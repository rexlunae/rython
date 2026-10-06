//! Regression tests pinning stdpython behavior to real Python semantics.

mod common;

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
    assert_eq!(py_floordiv(-7i64, 2).unwrap(), -4);
    assert_eq!(py_mod(-7i64, 2).unwrap(), 1);
    // Python: 7 // -2 == -4, 7 % -2 == -1
    assert_eq!(py_floordiv(7i64, -2).unwrap(), -4);
    assert_eq!(py_mod(7i64, -2).unwrap(), -1);
    // Positive operands match Rust.
    assert_eq!(py_floordiv(7i64, 2).unwrap(), 3);
    assert_eq!(py_mod(7i64, 2).unwrap(), 1);
    // Floats: -7.0 // 2.0 == -4.0
    assert_eq!(py_floordiv(-7.0f64, 2.0).unwrap(), -4.0);
    assert_eq!(py_mod(-7.0f64, 2.0).unwrap(), 1.0);
    // A zero divisor raises a catchable ZeroDivisionError (issue #75) —
    // no panic past try/except.
    assert_eq!(
        py_floordiv(1i64, 0).unwrap_err().exception_type,
        "ZeroDivisionError"
    );
    assert_eq!(py_mod(1.0f64, 0.0).unwrap_err().exception_type, "ZeroDivisionError");
}

#[test]
fn divmod_matches_python() {
    assert_eq!(divmod(-7i64, 2).unwrap(), (-4, 1));
    assert_eq!(divmod(7i64, 2).unwrap(), (3, 1));
    assert_eq!(divmod(1i64, 0).unwrap_err().exception_type, "ZeroDivisionError");
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
    assert_eq!(chr(97).unwrap(), "a");
    assert_eq!(chr(0x1F600).unwrap(), "😀");
    // Out-of-range raises the same ValueError as CPython; lone surrogates
    // raise a catchable ValueError (CPython succeeds, but UTF-8 cannot
    // represent them).
    assert_eq!(chr(-1).unwrap_err().exception_type, "ValueError");
    assert_eq!(chr(0x110000).unwrap_err().exception_type, "ValueError");
    assert_eq!(chr(0xD800).unwrap_err().exception_type, "ValueError");
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
    let mut obj = crate::PyDict::default();
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
fn math_fsum_compensates_and_matches_cpython() {
    // math.fsum (compensated summation). Every case pinned against
    // python3 3.14.
    use stdpython::math::fsum;
    // Empty sums to 0.0.
    assert_eq!(fsum(&[]).unwrap(), 0.0);
    // Error-free rounding: fsum([0.1, 0.2, 0.3]) is EXACTLY 0.6, where
    // the naive sum is 0.6000000000000001.
    assert_eq!(fsum(&[0.1, 0.2, 0.3]).unwrap(), 0.6);
    // A large term doesn't swallow a small one (python3: 1e-100).
    assert_eq!(fsum(&[1e100, -1e100, 1e-100]).unwrap(), 1e-100);
    assert_eq!(fsum(&[1e100, 1e-100, -1e100, 1e-100]).unwrap(), 2e-100);
    // An infinity short-circuits; NaN propagates.
    assert_eq!(fsum(&[1.0, f64::INFINITY]).unwrap(), f64::INFINITY);
    assert_eq!(fsum(&[1.0, f64::NEG_INFINITY]).unwrap(), f64::NEG_INFINITY);
    assert!(fsum(&[1.0, f64::NAN]).unwrap().is_nan());
    // An INTERMEDIATE overflow (a finite running total overflows to inf:
    // fsum([1e308, 1e308])) is OverflowError, exactly as in Python.
    let e = fsum(&[1.0f64, 1e308, 1e308]).unwrap_err();
    assert_eq!(
        format!("{}", e),
        "OverflowError: intermediate overflow in fsum"
    );
    // Half-even rounding across multiple partials: fsum([1e16, 1.0,
    // 1e-16]) rounds UP to 1.0000000000000002e16 (the 1e-16 makes the 1
    // slightly closer to 2); a naive sum would give 1e16.
    assert_eq!(fsum(&[1e16, 1.0, 1e-16]).unwrap(), 1.0000000000000002e16);
    assert_eq!(fsum(&[1e-16, 1.0, 1e16]).unwrap(), 1.0000000000000002e16);
    // Opposite infinities are ValueError (panics as the loud TypeError-
    // style error), matching CPython's "-inf + inf in fsum". A NaN
    // anywhere dominates an infinity.
    let e = fsum(&[f64::INFINITY, f64::NEG_INFINITY]).unwrap_err();
    assert_eq!(format!("{}", e), "ValueError: -inf + inf in fsum");
    let e = fsum(&[f64::NEG_INFINITY, f64::INFINITY]).unwrap_err();
    assert_eq!(format!("{}", e), "ValueError: -inf + inf in fsum");
    assert!(fsum(&[f64::INFINITY, f64::NAN]).unwrap().is_nan());
    assert!(fsum(&[f64::NAN, f64::INFINITY]).unwrap().is_nan());
}

#[test]
fn math_sumprod_preserves_int_and_float_types() {
    // math.sumprod (CPython 3.12+): the dot product. int×int stays int,
    // any float makes it float. Every case pinned against python3 3.14.
    use stdpython::math::{sumprod_f64, sumprod_i64};
    // All-int → int.
    assert_eq!(sumprod_i64(&[10, 20, 30], &[1, 2, 3]).unwrap(), 140);
    assert_eq!(sumprod_i64(&[1, 2], &[3, 4]).unwrap(), 11);
    // Empty → the additive identity (0).
    assert_eq!(sumprod_i64(&[], &[]).unwrap(), 0);
    // Float (at least one float element) → float.
    assert_eq!(sumprod_f64(&[1.5, 2.5], &[3.5, 4.5]).unwrap(), 16.5);
    assert_eq!(sumprod_f64(&[-1.0], &[1.0]).unwrap(), -1.0);
    // Unequal lengths raise ValueError (python3: `Inputs are not the same length`).
    let e = sumprod_i64(&[1, 2, 3], &[1, 2]).unwrap_err();
    assert_eq!(format!("{}", e), "ValueError: Inputs are not the same length");
}

#[test]
fn math_comb_perm_match_cpython() {
    // math.comb / math.perm (integer combinatorics), pinned against
    // python3 3.14.
    use stdpython::math::{comb, perm};
    assert_eq!(comb(5, 2).unwrap(), 10);
    assert_eq!(comb(10, 3).unwrap(), 120);
    assert_eq!(comb(5, 0).unwrap(), 1);
    assert_eq!(comb(0, 0).unwrap(), 1);
    // k > n → 0 (CPython), k < 0 → ValueError.
    assert_eq!(comb(5, 6).unwrap(), 0);
    let e = comb(5, -1).unwrap_err();
    assert_eq!(format!("{}", e), "ValueError: k must be a non-negative integer");
    // Negative n → ValueError.
    let e = comb(-1, 2).unwrap_err();
    assert_eq!(
        format!("{}", e),
        "ValueError: n must be a non-negative integer"
    );
    // A large coefficient that fits i64 but whose UNREDUCED recurrence
    // product overflows must not falsely raise (the gcd-cancelled
    // recurrence; python3: comb(62, 28) == 349615716557887465).
    assert_eq!(comb(62, 28).unwrap(), 349615716557887465);
    assert_eq!(perm(5, 2).unwrap(), 20);
    assert_eq!(perm(5, 3).unwrap(), 60);
    // k defaults to n at the codegen layer; here 2-arg is exact.
    assert_eq!(perm(5, 5).unwrap(), 120);
    assert_eq!(perm(0, 0).unwrap(), 1);
    assert_eq!(perm(5, 6).unwrap(), 0);
    let e = perm(5, -1).unwrap_err();
    assert_eq!(format!("{}", e), "ValueError: k must be a non-negative integer");
    assert_eq!(format!("{}", perm(-1, 2).unwrap_err()), "ValueError: n must be a non-negative integer");
}

#[test]
fn math_isqrt_cbrt_fma_hypot_nextafter_match_cpython() {
    // pinned against python3 3.14 (each assertion names its Python expression).
    use stdpython::math::{cbrt, fma, hypot, isqrt, nextafter};
    // Python: math.isqrt(16) == 4, .isqrt(15) == 3, .isqrt(0) == 0, .isqrt(1) == 1
    assert_eq!(isqrt(16).unwrap(), 4);
    assert_eq!(isqrt(15).unwrap(), 3);
    assert_eq!(isqrt(0).unwrap(), 0);
    assert_eq!(isqrt(1).unwrap(), 1);
    // Python: math.isqrt(9223372036854775807) == 3037000499 (i64::MAX, no
    // overflow in the Newton seeding — the (n+1)/2 ceiling wraps at MAX).
    for n in [i64::MAX, i64::MAX - 1, 99_999_999_999_999] {
        let r = isqrt(n).unwrap();
        assert_eq!(r, (n as f64).sqrt().floor() as i64, "math.isqrt({n})");
    }
    // Python: math.isqrt(-1) raises ValueError("isqrt() argument must be nonnegative")
    let e = isqrt(-1).unwrap_err();
    assert_eq!(format!("{}", e), "ValueError: isqrt() argument must be nonnegative");
    // cbrt routes through the Rust `libm` crate, a fixed correctly-rounded
    // software cube root (deterministic on every ABI): it returns the exact
    // root for perfect cubes (cbrt(-8.0) == -2.0, cbrt(0.0) == 0.0). CPython
    // instead calls the platform libm, which is NOT correctly rounded on
    // every system (glibc computes cbrt(27.0) == 3.0000000000000004 while
    // Apple's libm returns 3.0). So byte-exact cross-platform pins only hold
    // for inputs whose root is exact under every libm; cbrt(27.0) is bounded
    // within IEEE tolerance (a faithful-rounded cbrt is within 1 ulp).
    // Python: math.cbrt(-8.0) == -2.0, math.cbrt(0.0) == 0.0
    assert_eq!(cbrt(-8.0), -2.0);
    assert_eq!(cbrt(0.0), 0.0);
    // Python: abs(math.cbrt(27.0) - 3.0) < 1e-15
    assert!((cbrt(27.0) - 3.0).abs() < 1e-15, "cbrt(27.0) = {}", cbrt(27.0));
    // fma(x, y, z) = x*y + z with a single rounding; finite operands whose
    // exact product+sum overflows raise OverflowError, and a 0*inf/inf*0
    // (NaN from non-NaN inputs) raises ValueError — byte-identical to
    // CPython 3.13+ (math_helper returns the fused result otherwise).
    // Python: math.fma(3.0, 4.0, 5.0) == 17.0
    assert_eq!(fma(3.0, 4.0, 5.0).unwrap(), 17.0);
    // Python: math.fma(1e308, 1e308, 0.0) raises OverflowError("overflow in fma")
    assert_eq!(fma(1e308, 1e308, 0.0).unwrap_err().to_string(),
               "OverflowError: overflow in fma");
    // Python: math.fma(1e308, -1e308, 0.0) raises OverflowError("overflow in fma")
    assert_eq!(fma(1e308, -1e308, 0.0).unwrap_err().to_string(),
               "OverflowError: overflow in fma");
    // Python: math.fma(0.0, inf, 0.0) raises ValueError("invalid operation in fma")
    assert_eq!(fma(0.0, f64::INFINITY, 0.0).unwrap_err().to_string(),
               "ValueError: invalid operation in fma");
    // Python: math.fma(inf, 1.0, 0.0) == inf (no raise; infinite operand)
    assert_eq!(fma(f64::INFINITY, 1.0, 0.0).unwrap(), f64::INFINITY);
    // Python: math.fma(nan, 1.0, 0.0) is nan (no raise; NaN operand)
    assert!(fma(f64::NAN, 1.0, 0.0).unwrap().is_nan());
    // Python: math.hypot(3.0, 4.0) == 5.0, math.hypot(0.0, 0.0) == 0.0
    assert_eq!(hypot(3.0, 4.0), 5.0);
    assert_eq!(hypot(0.0, 0.0), 0.0);
    // Python: repr(math.nextafter(1.0, 2.0)) == '1.0000000000000002'
    assert_eq!(nextafter(1.0, 2.0), 1.0000000000000002);
    // Python: math.ulp(x) is the unit in the last place. ulp(±1.0) ==
    // sys.float_info.epsilon, ulp(0.0) == the min subnormal, ulp(2**52) ==
    // 1.0, ulp(inf) == inf, ulp(nan) == nan. Implemented as
    // `nextafter(|x|, +inf) - |x|` with inf/nan special cases.
    use stdpython::math::ulp;
    // Python: math.ulp(1.0) == sys.float_info.epsilon
    assert_eq!(ulp(1.0), f64::EPSILON);
    assert_eq!(ulp(-1.0), f64::EPSILON);
    // Python: math.ulp(0.0) == 5e-324 (the minimum positive subnormal)
    assert_eq!(ulp(0.0), f64::from_bits(1));
    // Python: math.ulp(2**52) == 1.0, math.ulp(2**53) == 2.0
    assert_eq!(ulp(4503599627370496.0), 1.0);
    assert_eq!(ulp(9007199254740992.0), 2.0);
    // Python: math.ulp(inf) == inf, math.ulp(nan) is nan
    assert_eq!(ulp(f64::INFINITY), f64::INFINITY);
    assert!(ulp(f64::NAN).is_nan());
    // Python: math.ulp(1.7976931348623157e308) == 2.0**971 (the finite
    // predecessor gap at the largest double — nextafter(|MAX|,+inf) would
    // wrongly give inf).
    assert_eq!(ulp(f64::MAX), 2.0f64.powi(971));
}

#[test]
fn math_log_one_and_two_arg_match_python() {
    // Python: math.log(e) == 1.0 (base None), math.log(8, 2) == 3.0,
    // math.log(100, 10) == 2.0; a non-positive x or invalid base raises
    // ValueError("math domain error").
    use stdpython::math::log;
    assert_eq!(log(2.718281828459045, None).unwrap(), 1.0);
    assert_eq!(log(8.0, Some(2.0)).unwrap(), 3.0);
    assert_eq!(log(100.0, Some(10.0)).unwrap(), 2.0);
    assert_eq!(format!("{}", log(-1.0, None).unwrap_err()), "ValueError: math domain error");
    assert_eq!(format!("{}", log(8.0, Some(1.0)).unwrap_err()), "ValueError: math domain error");
}

#[test]
fn math_gamma_matches_python_values_and_pole() {
    // Python: math.gamma(5.0)==24.0, gamma(0.5)≈1.7724538509055159,
    // gamma(-2.5)≈-0.9453087204829418, gamma(1.0)==1.0; a 0 / non-positive
    // integer is a pole (ValueError); a finite input whose gamma overflows
    // raises OverflowError("math range error"). lgamma(0.5) is within 1 ulp
    // of CPython (libm software).
    use stdpython::math::{gamma, lgamma};
    assert_eq!(gamma(5.0).unwrap(), 24.0);
    assert_eq!(gamma(1.0).unwrap(), 1.0);
    // Non-integer gamma goes through CPython's platform lgamma and can be
    // 1 ulp off across libms (macOS vs glibc), so pin it within tolerance.
    assert!((gamma(0.5).unwrap() - 1.7724538509055159).abs() < 1e-12);
    assert!((gamma(-2.5).unwrap() - (-0.9453087204829418)).abs() < 1e-12);
    assert_eq!(format!("{}", gamma(0.0).unwrap_err()), "ValueError: math domain error");
    assert_eq!(format!("{}", gamma(-3.0).unwrap_err()), "ValueError: math domain error");
    assert_eq!(
        format!("{}", gamma(171.7).unwrap_err()),
        "OverflowError: math range error"
    );
    // Python: math.lgamma(0.5) ≈ 0.5723649429247004 (within tolerance), and
    // lgamma(-inf) == +inf (NOT a pole), while a large FINITE input raises
    // OverflowError("math range error") (lgamma(1e308)).
    assert!((lgamma(0.5).unwrap() - 0.5723649429247004).abs() < 1e-12);
    assert_eq!(lgamma(f64::NEG_INFINITY).unwrap(), f64::INFINITY);
    assert_eq!(format!("{}", lgamma(1e308).unwrap_err()), "OverflowError: math range error");
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

    // Python: "bo" in ["ada", "bo"] — a str probe tests a container of
    // OWNED strings by content (issue #229: the class-field shapes reach
    // the trait with a literal operand). Both the &&str the renderer
    // emits and the &str spelling must hold.
    assert!(names.py_contains(&"bo"));
    assert!(!names.py_contains(&"zz"));
    assert!(names.py_contains("bo"));
    assert!(!names.py_contains("zz"));

    // Python: "k" in {"k": 1} — a str probe on a String-keyed dict
    let sd = PyDict::<String, i64>::from([("k".to_string(), 1i64)]);
    assert!(sd.py_contains(&"k"));
    assert!(!sd.py_contains(&"z"));
    let shm = std::collections::HashMap::from([("k".to_string(), 1i64)]);
    assert!(shm.py_contains(&"k"));
    assert!(!shm.py_contains(&"z"));

    // Python: "x" in {"x"} — a str probe on a set of owned strings
    let ss = std::collections::HashSet::from([String::from("x")]);
    assert!(ss.py_contains(&"x"));
    assert!(!ss.py_contains(&"y"));
    assert!(ss.py_contains("x"));
    assert!(!ss.py_contains("y"));

    // Python: 2 in {1, 2, 3} — set literals lower to a std HashSet
    let s = std::collections::HashSet::from([1i64, 2, 3]);
    assert!(s.py_contains(&2));
    assert!(!s.py_contains(&9));
}

#[test]
fn boxed_pyvalue_containment_matches_python() {
    // Python: "a" in "abc" / "z" in "abc" — substring on the str member.
    assert!(PyValue::from("abc").py_contains(&PyValue::from("a")));
    assert!(!PyValue::from("abc").py_contains(&PyValue::from("z")));

    // Python: 1 in (1, 2.0) — element equality is ==, so numeric kinds
    // compare by value across int/float/bool (the derived PartialEq
    // would say False for 1 == 1.0).
    let t = PyValue::Tuple(std::sync::Arc::new(vec![
        PyValue::Int(1),
        PyValue::Float(2.0),
    ]));
    assert!(t.py_contains(&PyValue::Int(1)));
    assert!(t.py_contains(&PyValue::Float(1.0)));
    assert!(!t.py_contains(&PyValue::Int(9)));

    // Python: "k" in {"k": 1} — key lookup on the dict member; a non-str
    // member is never a key.
    let d = PyValue::Dict(std::sync::Arc::new(PyDict::from([(
        "k".to_string(),
        PyValue::Int(1),
    )])));
    assert!(d.py_contains(&PyValue::from("k")));
    assert!(!d.py_contains(&PyValue::from("z")));
    assert!(!d.py_contains(&PyValue::Int(1)));

    // Python: b"a" in b"abc" (subsequence), 97 in b"abc" (octet).
    let b = PyValue::Bytes(b"abc".to_vec());
    assert!(b.py_contains(&PyValue::Bytes(b"a".to_vec())));
    assert!(b.py_contains(&PyValue::Bytes(Vec::new())));
    assert!(!b.py_contains(&PyValue::Bytes(b"z".to_vec())));
    assert!(b.py_contains(&PyValue::Int(97)));
    assert!(!b.py_contains(&PyValue::Int(122)));

    // A str/String probe reaches the same semantics through the
    // renderer's owned spellings (`k in boxed` where k is a String).
    assert!(PyValue::from("abc").py_contains(&"ab".to_string()));
    assert!(PyValue::from("abc").py_contains(&"a"));
    assert!(!PyValue::from("abc").py_contains(&"zz".to_string()));
}

#[test]
#[should_panic(expected = "TypeError: 'in <string>' requires string as left operand, not int")]
fn boxed_string_containment_rejects_int_like_cpython() {
    // Python 3.11: 1 in "abc" raises TypeError — CPython's exact text.
    let _ = PyValue::from("abc").py_contains(&PyValue::Int(1));
}

#[test]
#[should_panic(expected = "TypeError: argument of type 'int' is not iterable")]
fn boxed_int_containment_is_a_type_error_like_cpython() {
    // Python 3.11: 1 in 5 raises TypeError — CPython's exact text.
    let _ = PyValue::Int(5).py_contains(&PyValue::Int(1));
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
fn py_exception_type_name_is_the_raised_class() {
    // type(ValueError()).__name__ -> 'ValueError'
    assert_eq!(PyException::new("ValueError", "").type_name(), "ValueError");
    // type(IOError()).__name__ -> 'OSError' (the alias IS the class);
    // likewise EnvironmentError
    assert_eq!(PyException::new("IOError", "").type_name(), "OSError");
    assert_eq!(PyException::new("EnvironmentError", "").type_name(), "OSError");
    // A user class (class IDNAError(UnicodeError)) keeps its own name.
    assert_eq!(PyException::new("IDNAError", "").type_name(), "IDNAError");
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
fn issue81_string_divergences_match_cpython() {
    // splitlines: full Python boundary set (\r \v \f \x1c-\x1e \x85
    // \u2028 \u2029), \r\n as ONE boundary, no trailing empty line, but
    // empties between consecutive boundaries.
    assert_eq!("a\rb".splitlines(), vec!["a", "b"]);
    assert_eq!("a\u{c}b".splitlines(), vec!["a", "b"]);
    assert_eq!("a\u{1d}b".splitlines(), vec!["a", "b"]);
    assert_eq!("a\u{85}b".splitlines(), vec!["a", "b"]);
    assert_eq!("a\u{2028}b".splitlines(), vec!["a", "b"]);
    assert_eq!("a\r\nb".splitlines(), vec!["a", "b"]);
    assert_eq!("a\n\n".splitlines(), vec!["a", ""]);
    assert_eq!("\n".splitlines(), vec![""]);
    assert_eq!("".splitlines(), Vec::<String>::new());
    assert_eq!("a\n".splitlines(), vec!["a"]);

    // capitalize/title use TITLECASE, not uppercase, for the first letter.
    assert_eq!("\u{fb01}le".capitalize(), "File"); // ﬁ -> Fi
    assert_eq!("\u{00df}".capitalize(), "Ss"); // ß -> Ss
    assert_eq!("\u{01f3}".title(), "\u{01f2}"); // ǳ -> ǲ
    assert_eq!("3rd".title(), "3Rd");

    // swapcase toggles each cased char's CASE — the UPPERCASE expansion,
    // not titlecase: ß -> SS, ǆ -> Ǆ, ﬃ -> FFI (verified against python3).
    assert_eq!("The Quick".swapcase(), "tHE qUICK");
    assert_eq!("\u{00df}".swapcase(), "SS"); // ß -> SS
    assert_eq!("\u{01c6}".swapcase(), "\u{01c4}"); // ǆ -> Ǆ
    assert_eq!("\u{fb03}".swapcase(), "FFI"); // ﬃ -> FFI

    // The \x1c-\x1f separators are whitespace for strip/split.
    assert_eq!("hello\u{1f}".strip(), "hello");
    assert_eq!("a\u{1c}b".py_split_whitespace(), vec!["a", "b"]);

    // repr escapes everything str.isprintable() rejects.
    assert_eq!(py_str_repr("\u{a0}"), "'\\xa0'");
    assert_eq!(py_str_repr("\u{ad}"), "'\\xad'");
    assert_eq!(py_str_repr("\u{200b}"), "'\\u200b'");
    assert_eq!(py_str_repr("\u{2028}"), "'\\u2028'");
    assert_eq!(py_str_repr("plain"), "'plain'");
}

#[test]
fn issue81_round_and_pow_match_cpython() {
    // round(x, n): half-even at the correctly-rounded decimal, verified
    // against python3 over a 46-value × 9-ndigits sweep.
    assert_eq!(round_digits(1.15, 1), 1.1);
    assert_eq!(round_digits(2.675, 2), 2.67);
    assert_eq!(round_digits(0.005, 2), 0.01); // 0.005 is stored slightly ABOVE
    assert_eq!(round_digits(2.5, 0), 2.0);
    assert_eq!(round_digits(3.5, 0), 4.0);
    assert_eq!(round_digits(1250.0, -2), 1200.0);
    assert_eq!(round_digits(1350.0, -2), 1400.0);
    assert_eq!(round_digits(1234.5, -2), 1200.0);
    assert_eq!(round_digits(15.0, -1), 20.0);
    assert_eq!(round_digits(5.0, -1), 0.0);
    assert_eq!(round_digits(1e308, -400), 0.0);
    assert_eq!(round_digits(f64::INFINITY, 2), f64::INFINITY);

    // i64 ** huge exponent: no u32 truncation (0 ** 4294967296 is 0, 1
    // is 1, (-1) ** odd is -1); anything else overflows loudly.
    assert_eq!(py_pow(0i64, 4294967296i64), 0);
    assert_eq!(py_pow(1i64, 4294967296i64), 1);
    assert_eq!(py_pow(-1i64, 4294967297i64), -1);
    assert!(std::panic::catch_unwind(|| py_pow(2i64, 4294967296i64)).is_err());
}

#[test]
fn issue81_negative_index_overflow_does_not_panic() {
    // i64::MIN + len used to overflow in normalize_index; Python answers
    // IndexError because |index| > len.
    assert!(vec![1i64, 2]
        .py_index(-9223372036854775808i64)
        .is_err());
    assert!(vec![1i64, 2]
        .py_pop(-9223372036854775808i64)
        .is_err());
    // insert prepends, like Python (insert(-huge, x) == insert(0, x)).
    let mut v = vec![1i64, 2];
    let _ = v.py_insert(-9223372036854775808i64, 9);
    assert_eq!(v, vec![9, 1, 2]);
}

#[test]
fn py_insert_matches_python_index_rules() {
    // Python: [1, 2, 3].insert(-1, 9) -> [1, 2, 9, 3]
    let mut v = vec![1i64, 2, 3];
    let _ = v.py_insert(-1, 9);
    assert_eq!(v, vec![1, 2, 9, 3]);
    // insert(100, x) clamps to append
    let mut v = vec![1i64, 2];
    let _ = v.py_insert(100, 9);
    assert_eq!(v, vec![1, 2, 9]);
    // insert(-100, x) clamps to prepend
    let mut v = vec![1i64, 2];
    let _ = v.py_insert(-100, 9);
    assert_eq!(v, vec![9, 1, 2]);
    // plain positive index
    let mut v = vec![1i64, 3];
    let _ = v.py_insert(1, 2);
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
    let dir = common::create_scratch("glob-hidden");
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

#[test]
fn tempfile_gettempdir_picks_first_usable_cpython_candidate() {
    // Python: tempfile.gettempdir() probes $TMPDIR, $TEMP, $TMP, /tmp,
    // /var/tmp, /usr/tmp and finally os.getcwd(), returning the first
    // directory that accepts a created-and-removed file. Verified against
    // python3 3.14 on macOS: with every temp variable unset it returns
    // '/tmp' — not Rust's Darwin-specific /var/folders/... path — and even
    // TMPDIR=/nonexistent-xyz still yields '/tmp'.
    let dir = stdpython::stdlib::tempfile::gettempdir();
    assert!(dir.is_absolute(), "gettempdir must be absolute: {:?}", dir);
    let probe = dir.join(format!(
        "rython-tempfile-pin-{}",
        std::process::id()
    ));
    std::fs::write(&probe, b"x")
        .expect("gettempdir must return a directory that accepts writes");
    let _ = std::fs::remove_file(&probe);
}

#[test]
fn heterogeneous_union_values_str_repr_and_display() {
    // Issue #121: str | bytes lowers to StrOrBytes and wider unions to
    // the boxed PyValue. Python str()/repr() semantics, verified against
    // python3 3.14:
    //   str('s')=='s'; str(b'raw')=="b'raw'"; str(b"a'b")=='b"a\'b"'
    //   (bytes repr switches to double quotes); non-printable bytes are
    //   \xNN lowercase; str(7)=='7'; str(True)=='True'; str(3.5)=='3.5';
    //   str(None)=='None'; str((1,'a'))=="(1, 'a')"; str((1,))=='(1,)';
    //   repr('s')=="'s'".
    use stdpython::{
        py_bytes_repr, py_value_repr, py_value_str, PyValue, StrOrBytes,
    };
    let sb = |v: StrOrBytes| v.py_str();
    assert_eq!(sb(StrOrBytes::from("s")), "s");
    assert_eq!(sb(StrOrBytes::from(b"raw".as_slice())), "b'raw'");
    assert_eq!(
        sb(StrOrBytes::Bytes(vec![b'a', b'\'', b'b'])),
        "b\"a'b\""
    );
    assert_eq!(
        py_bytes_repr(&[0x00, 0x7f, 0x80, 0xff]),
        "b'\\x00\\x7f\\x80\\xff'"
    );

    let pv = |v: &PyValue| py_value_str(v);
    assert_eq!(pv(&PyValue::Int(7)), "7");
    assert_eq!(pv(&PyValue::Float(3.5)), "3.5");
    assert_eq!(pv(&PyValue::Bool(true)), "True");
    assert_eq!(pv(&PyValue::Bool(false)), "False");
    assert_eq!(pv(&PyValue::Str("s".into())), "s");
    assert_eq!(
        pv(&PyValue::Bytes(vec![b'r', b'a', b'w'])),
        "b'raw'"
    );
    assert_eq!(pv(&PyValue::None_), "None");
    assert_eq!(
        pv(&PyValue::Tuple(std::sync::Arc::new(vec![
            PyValue::Int(1),
            PyValue::Str("a".into()),
        ]))),
        "(1, 'a')"
    );
    assert_eq!(
        pv(&PyValue::Tuple(std::sync::Arc::new(vec![PyValue::Int(1)]))),
        "(1,)"
    );
    // A boxed dict (issue #180): str renders like a Python dict, and
    // indexing a boxed dict dispatches to the held dict — anything else
    // raises the TypeError.
    let mut boxed: stdpython::PyDict<String, PyValue> = stdpython::PyDict::new();
    boxed.insert("ProviderType".into(), PyValue::Str("sso".into()));
    boxed.insert("n".into(), PyValue::Int(7));
    let pv_dict = PyValue::Dict(std::sync::Arc::new(boxed));
    assert_eq!(pv(&pv_dict), "{'ProviderType': 'sso', 'n': 7}");
    assert_eq!(pv(&pv_dict.py_index("ProviderType").unwrap()), "sso");
    assert!(pv_dict.py_index("missing").is_err());
    let not_dict = PyValue::Int(1);
    assert!(not_dict.py_index("x").is_err());
    // iterating a boxed dict yields its KEYS (Python semantics).
    let keys: Vec<PyValue> = pv_dict.clone().into_iter().collect();
    assert_eq!(
        keys.iter().map(|k| pv(k)).collect::<Vec<_>>(),
        vec!["ProviderType", "n"]
    );

    // repr quotes strs; every other member matches its str form.
    assert_eq!(py_value_repr(&PyValue::Str("s".into())), "'s'");
    assert_eq!(py_value_repr(&PyValue::Int(7)), "7");

    // Boxed floats as KEYS: 0.0 and -0.0 are the SAME key in Python
    // ({0.0: 'a'}[-0.0] == 'a' - verified against python3 3.14), so their
    // hashes must agree even though the bit patterns differ.
    use std::collections::HashSet;
    let mut keys = HashSet::new();
    keys.insert(PyValue::Float(0.0));
    assert!(keys.contains(&PyValue::Float(-0.0)));
    keys.insert(PyValue::Float(-0.0));
    assert_eq!(keys.len(), 1);

    // print() uses the same rendering as str().
    use stdpython::PyDisplay;
    assert_eq!(PyDisplay::py_display(&StrOrBytes::from("s")), "s");
    assert_eq!(
        PyDisplay::py_display(&PyValue::Bytes(vec![b'r', b'a', b'w'])),
        "b'raw'"
    );
}

#[test]
fn exception_matching_walks_the_cpython_hierarchy() {
    // Python: `except X:` catches raised type E iff X is E or an ancestor
    // of E. Verified against python3 3.14 by raising each leaf under its
    // claimed clause (and confirming the negatives escape):
    //   FileNotFoundError IS-A OSError, Exception, BaseException,
    //     EnvironmentError (the OSError alias)
    //   KeyError / IndexError ARE-A LookupError, but NOT each other
    //   UnicodeDecodeError IS-A ValueError (two hops via UnicodeError)
    //   TabError IS-A SyntaxError (three hops)
    //   ZeroDivisionError IS-A ArithmeticError
    //   SystemExit / KeyboardInterrupt / GeneratorExit are NOT caught by
    //     `except Exception:` (they hang off BaseException directly)
    use stdpython::PyException;
    let file_nf = PyException::new("FileNotFoundError", "gone");
    assert!(file_nf.matches("FileNotFoundError"));
    assert!(file_nf.matches("OSError"));
    assert!(file_nf.matches("EnvironmentError"));
    assert!(file_nf.matches("Exception"));
    assert!(file_nf.matches("BaseException"));

    let key_err = PyException::new("KeyError", "k");
    assert!(key_err.matches("LookupError"));
    assert!(!PyException::new("IndexError", "i").matches("KeyError"));

    assert!(
        PyException::new("UnicodeDecodeError", "bad").matches("ValueError"),
        "two-hop ancestry through UnicodeError"
    );
    assert!(
        PyException::new("TabError", "tabs").matches("SyntaxError"),
        "three-hop ancestry"
    );
    assert!(
        PyException::new("ZeroDivisionError", "/0").matches("ArithmeticError")
    );

    for base_only in ["SystemExit", "KeyboardInterrupt", "GeneratorExit"] {
        assert!(
            !PyException::new(base_only, "").matches("Exception"),
            "{base_only} must not be caught by except Exception:"
        );
        assert!(PyException::new(base_only, "").matches("BaseException"));
    }

    // BaseException is the tree's root: exact catch only.
    // Verified against python3 3.14: raising BaseException under
    // `except Exception:` escapes; `except BaseException:` catches it.
    let root = PyException::new("BaseException", "");
    assert!(!root.matches("Exception"));
    assert!(root.matches("BaseException"));

    // ExceptionGroup multiply inherits (BaseExceptionGroup, Exception):
    // `except Exception:` catches it, but a bare BaseExceptionGroup stays
    // outside Exception. Verified against python3 3.14 with
    // ExceptionGroup('eg', [ValueError('v')]) raised under each clause.
    let eg = PyException::new("ExceptionGroup", "eg");
    assert!(eg.matches("Exception"));
    assert!(eg.matches("BaseExceptionGroup"));
    assert!(eg.matches("BaseException"));
    assert!(!eg.matches("ValueError"));
    let bg = PyException::new("BaseExceptionGroup", "bg");
    assert!(bg.matches("BaseExceptionGroup"));
    assert!(!bg.matches("Exception"));

    assert!(!key_err.matches("TypeError"), "siblings do not catch");
}

#[test]
fn builtin_discriminant_matching_walks_the_same_cpython_hierarchy() {
    // Round 52: `matches_builtin` is the discriminant fast path generated
    // code emits for literal `except <builtin>:` clauses. It must agree
    // with the string `matches` walk exactly — same interpreter-derived
    // MRO, integer comparison instead of string search. Every assertion
    // here mirrors one in
    // `exception_matching_walks_the_cpython_hierarchy` (same CPython
    // 3.14 verification).
    use stdpython::{BuiltinException as B, PyException};
    let file_nf = PyException::new("FileNotFoundError", "gone");
    assert!(file_nf.matches_builtin(B::FileNotFoundError));
    assert!(file_nf.matches_builtin(B::OSError));
    assert!(file_nf.matches_builtin(B::Exception));
    assert!(file_nf.matches_builtin(B::BaseException));
    assert!(!file_nf.matches_builtin(B::KeyError), "siblings do not catch");

    let key_err = PyException::new("KeyError", "k");
    assert!(key_err.matches_builtin(B::LookupError));
    assert!(!PyException::new("IndexError", "i").matches_builtin(B::KeyError));

    assert!(
        PyException::new("UnicodeDecodeError", "bad").matches_builtin(B::ValueError),
        "two-hop ancestry through UnicodeError"
    );
    assert!(PyException::new("TabError", "tabs").matches_builtin(B::SyntaxError));

    for base_only in ["SystemExit", "KeyboardInterrupt", "GeneratorExit"] {
        assert!(
            !PyException::new(base_only, "").matches_builtin(B::Exception),
            "{base_only} must not be caught by except Exception:"
        );
        assert!(PyException::new(base_only, "").matches_builtin(B::BaseException));
    }

    // A raised USER class has no discriminant: only the broad posture
    // (same as matches()).
    let user = PyException::new("MyError", "u");
    assert!(user.matches_builtin(B::Exception));
    assert!(user.matches_builtin(B::BaseException));
    assert!(!user.matches_builtin(B::ValueError));
}

#[test]
fn exception_leaf_constructors_carry_their_type_names() {
    use stdpython::*;
    assert_eq!(import_error("no module").exception_type, "ImportError");
    assert_eq!(
        module_not_found_error("nomod").exception_type,
        "ModuleNotFoundError"
    );
    assert_eq!(stop_iteration("").exception_type, "StopIteration");
    assert_eq!(recursion_error("deep").exception_type, "RecursionError");
    assert_eq!(
        unicode_decode_error("bad bytes").exception_type,
        "UnicodeDecodeError"
    );
    assert_eq!(timeout_error("slow").exception_type, "TimeoutError");
    // The hierarchy sees through every constructor.
    assert!(file_not_found_error("x").matches("OSError"));
    assert!(module_not_found_error("x").matches("ImportError"));
}

#[test]
fn ascii_escapes_outside_printable_ascii_like_python() {
    // Python: ascii() is repr() with non-printable-ASCII code points
    // escaped (\xXX, \uXXXX, \UXXXXXXXX — lowercase hex). Verified against
    // python3 3.14:
    //   ascii('café')      -> "'caf\\xe9'"
    //   ascii('😀')         -> "'\\U0001f600'"
    //   ascii(42)          -> '42'
    //   ascii(True)        -> 'True'
    //   ascii(3.5)         -> '3.5'
    //   ascii('\n')        -> "'\\n'"   (repr already escapes it)
    //   ascii(chr(0x7f))   -> "'\\x7f'" (DEL escapes; printable passes)
    use stdpython::ascii;
    assert_eq!(ascii("café"), "'caf\\xe9'");
    assert_eq!(ascii("😀"), "'\\U0001f600'");
    assert_eq!(ascii("\n"), "'\\n'");
    assert_eq!(ascii("\u{7f}"), "'\\x7f'");
    assert_eq!(ascii(&42i64), "42");
    assert_eq!(ascii(&true), "True");
    assert_eq!(ascii(&3.5f64), "3.5");
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
        let m = re::search(r"(\d+)-(\d+)", "order 12-34 shipped", "").unwrap();
        assert_eq!(m.group(0), "12-34");
        assert_eq!(m.group(1), "12");
        assert_eq!(m.group(2), "34");
        assert_eq!(m.groups(), vec!["12", "34"]);
        assert_eq!((m.start(), m.end()), (6, 11));
        assert_eq!(m.span(), (6, 11));

        // match anchors at the start; fullmatch also at the end.
        assert!(re::r#match(r"\d+", "12ab", "").unwrap().is_some());
        assert!(re::r#match(r"\d+", "ab12", "").unwrap().is_none());
        assert!(re::fullmatch(r"\d+", "123", "").unwrap().is_some());
        assert!(re::fullmatch(r"\d+", "123a", "").unwrap().is_none());
    }

    #[test]
    fn offsets_are_character_offsets_like_python() {
        // python3: re.search(r"héllo", "say héllo").span() == (4, 9) —
        // characters, not the regex crate's bytes.
        let m = re::search("héllo", "say héllo", "").unwrap();
        assert_eq!(m.span(), (4, 9));
    }

    #[test]
    fn findall_sub_split_match_python() {
        assert_eq!(
            re::findall(r"\d+", "a1 b22 c333", "").unwrap(),
            vec!["1", "22", "333"]
        );
        // One capture group: findall yields the group.
        assert_eq!(re::findall(r"(\w)\d", "a1 b2", "").unwrap(), vec!["a", "b"]);
        assert_eq!(re::findall("x", "abc", "").unwrap(), Vec::<String>::new());
        // Two-plus groups yield tuples in Python: loud, not wrong-shaped.
        assert!(re::findall(r"(a)(b)", "ab", "").is_err());

        assert_eq!(re::sub(r"(\d+)", r"<\1>", "a1 b22", 0, "").unwrap(), "a<1> b<22>");
        assert_eq!(re::sub("cat", "dog", "cat cat", 0, "").unwrap(), "dog dog");

        assert_eq!(
            re::split(r"[,;]\s*", "a, b;c ,d", 0, "").unwrap(),
            vec!["a", "b", "c ", "d"]
        );
        assert_eq!(re::split(r"\d", "abc", 0, "").unwrap(), vec!["abc"]);
        // Capturing groups interleave the delimiters, as in Python:
        // re.split(r"(\d)", "a1b") == ['a', '1', 'b'].
        assert_eq!(re::split(r"(\d)", "a1b", 0, "").unwrap(), vec!["a", "1", "b"]);
        assert_eq!(re::split(r"(\d)", "1", 0, "").unwrap(), vec!["", "1", ""]);
        assert_eq!(
            re::split(r"([,;])\s*", "a, b;c", 0, "").unwrap(),
            vec!["a", ",", "b", ";", "c"]
        );
        // A non-participating group becomes None in Python — loud here.
        let e = re::split(r"(x)|(\d)", "a1b", 0, "").unwrap_err();
        assert!(
            format!("{}", e).contains("did not participate"),
            "err: {}",
            e
        );
    }

    #[test]
    fn errors_are_loud() {
        // Unsupported-by-the-engine patterns (Python allows lookbehind)
        // and bad patterns both fail as re.error.
        let e = re::search(r"(?<=a)b(", "x", "").unwrap_err();
        assert!(format!("{}", e).starts_with("re.error:"), "err: {}", e);

        // A missed match behaves like Python's None.group(): loud
        // AttributeError with Python's message.
        let miss = re::search(r"\d", "abc", "").unwrap();
        assert!(miss.is_none());
        let result = std::panic::catch_unwind(|| miss.group(0));
        let msg = *result.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(
            msg,
            "AttributeError: 'NoneType' object has no attribute 'group'"
        );
    }

    #[test]
    fn flags_count_and_finditer_match_python() {
        // python3: findall(r"ab", "AB ab Ab", re.IGNORECASE)
        assert_eq!(
            re::findall("ab", "AB ab Ab", "i").unwrap(),
            vec!["AB", "ab", "Ab"]
        );
        assert_eq!(re::findall("^x", "x\nyx\nxz", "m").unwrap(), vec!["x", "x"]);
        assert_eq!(
            re::findall("a.b", "a\nb axb", "s").unwrap(),
            vec!["a\nb", "axb"]
        );
        assert_eq!(re::findall("^a.b$", "A\nB", "ims").unwrap(), vec!["A\nB"]);

        // sub count: 0 replaces all, positive limits, negative none.
        assert_eq!(re::sub("a", "-", "aaaa", 2, "").unwrap(), "--aa");
        assert_eq!(re::sub("a", "-", "aaaa", 0, "").unwrap(), "----");
        assert_eq!(re::sub("a", "-", "aaaa", -1, "").unwrap(), "aaaa");
        // A bad pattern raises even when the negative count/maxsplit
        // means no work would happen, as in Python.
        assert!(re::sub("(", "-", "aaaa", -1, "").is_err());
        assert!(re::split("(", "aaaa", -1, "").is_err());

        // split maxsplit: 0 unlimited, positive limits, negative none.
        assert_eq!(
            re::split(r"\s+", "a b c d", 1, "").unwrap(),
            vec!["a", "b c d"]
        );
        assert_eq!(
            re::split(r"\s+", "a b c d", -1, "").unwrap(),
            vec!["a b c d"]
        );
        // Capturing groups still interleave under a limit.
        assert_eq!(
            re::split(r"(\d)", "a1b2c", 1, "").unwrap(),
            vec!["a", "1", "b2c"]
        );
        assert_eq!(
            re::split("x", "AxBxC", 1, "i").unwrap(),
            vec!["A", "BxC"]
        );

        let spans: Vec<(i64, i64)> = re::finditer(r"\d+", "a1 b22", "")
            .unwrap()
            .iter()
            .map(|m| m.span())
            .collect();
        assert_eq!(spans, vec![(1, 2), (4, 6)]);
        let groups: Vec<String> = re::finditer(r"\d+", "a1 b22", "")
            .unwrap()
            .iter()
            .map(|m| m.group(0))
            .collect();
        assert_eq!(groups, vec!["1", "22"]);
    }

    #[test]
    #[should_panic(expected = "no such group")]
    fn out_of_range_groups_raise_index_error() {
        let m = re::search("a", "a", "").unwrap();
        let _ = m.group(3);
    }

    #[test]
    #[should_panic(expected = "did not participate")]
    fn non_participating_groups_fail_loudly() {
        // python3 returns None for group(2) of r"(a)(b)?" on "a"; a typed
        // String cannot, so this is loud instead of invented.
        let m = re::search(r"(a)(b)?", "a", "").unwrap();
        let _ = m.group(2);
    }
}

// ---------------------------------------------------------------------------
// map/filter/list builtins (issue #19). Pinned against python3.
// ---------------------------------------------------------------------------

mod map_filter_list {
    use stdpython::*;

    #[test]
    fn map_and_filter_match_python() {
        assert_eq!(map(|x: i64| x * 2, vec![1, 2, 3]), vec![2, 4, 6]);
        // Two iterables pair up to the shortest, like zip.
        assert_eq!(
            map2(|a: i64, b: i64| a + b, vec![1, 2], vec![10, 20, 30]),
            vec![11, 22]
        );
        assert_eq!(filter(|x: i64| x > 1, vec![1, 2, 3]), vec![2, 3]);
        assert_eq!(filter_truthy(vec![0i64, 3, 0, 5]), vec![3, 5]);

        // The fallible forms propagate the first exception.
        let ok = map_fallible(|x: i64| Ok(x + 1), vec![1, 2]).unwrap();
        assert_eq!(ok, vec![2, 3]);
        let err = map_fallible(
            |x: i64| {
                if x == 2 {
                    Err(value_error("bad"))
                } else {
                    Ok(x)
                }
            },
            vec![1, 2, 3],
        )
        .unwrap_err();
        assert_eq!(format!("{}", err), "ValueError: bad");
        let kept =
            filter_fallible(|x: i64| Ok(x % 2 == 0), vec![1, 2, 3, 4]).unwrap();
        assert_eq!(kept, vec![2, 4]);
    }

    #[test]
    fn list_builtin_follows_pythons_shapes() {
        assert_eq!(list(vec![1i64, 2]), vec![1, 2]);
        // list("ab") explodes into one-character strings.
        assert_eq!(list("ab"), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(list(range(3)), vec![0i64, 1, 2]);
    }
}

// ---------------------------------------------------------------------------
// hashlib (issue #19). Digests pinned against python3.
// ---------------------------------------------------------------------------

mod hashlib_module {
    use stdpython::hashlib::*;

    #[test]
    fn digests_match_cpython_exactly() {
        assert_eq!(md5("hello").hexdigest(), "5d41402abc4b2a76b9719d911017c592");
        assert_eq!(
            sha1("hello").hexdigest(),
            "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
        );
        assert_eq!(
            sha256("hello").hexdigest(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert!(sha512("hello")
            .hexdigest()
            .starts_with("9b71d224bd62f3785d96d46ad3ea3d73"));
        // Non-ASCII hashes its UTF-8 bytes: hashlib.md5("café".encode()).
        assert_eq!(md5("café").hexdigest(), "07117fe4a1ebd544965dc19573183da2");
        assert_eq!(
            sha256("").hexdigest(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn update_accumulates_and_hexdigest_does_not_consume() {
        let mut h = sha256_new();
        h.update("hel");
        h.update("lo");
        assert_eq!(h.hexdigest(), sha256("hello").hexdigest());
        // hexdigest() again (Python allows further updates after reading).
        assert_eq!(h.hexdigest(), sha256("hello").hexdigest());
        h.update("!");
        assert_eq!(h.hexdigest(), sha256("hello!").hexdigest());
    }
}

// ---------------------------------------------------------------------------
// textwrap.wrap/fill (issue #19). All expected values pinned against
// python3 with the default settings.
// ---------------------------------------------------------------------------

mod textwrap_wrap {
    use stdpython::textwrap::{fill, wrap};

    #[test]
    fn wrapping_matches_python() {
        assert_eq!(
            wrap("The quick brown fox jumps over the lazy dog", 10).unwrap(),
            vec!["The quick", "brown fox", "jumps over", "the lazy", "dog"]
        );
        // Hyphenated words break after acceptable hyphens.
        assert_eq!(
            wrap("a self-referential well-known example", 12).unwrap(),
            vec!["a self-", "referential", "well-known", "example"]
        );
        // Em-dashes between words are their own chunks.
        assert_eq!(
            wrap("hello--world and then--some", 8).unwrap(),
            vec!["hello--", "world", "and then", "--some"]
        );
        assert_eq!(
            wrap("word wrap-ping is--neat", 6).unwrap(),
            vec!["word", "wrap-", "ping", "is--", "neat"]
        );
        assert_eq!(fill("one two three four", 9).unwrap(), "one two\nthree\nfour");
    }

    #[test]
    fn long_words_break_like_python() {
        assert_eq!(
            wrap("supercalifragilisticexpialidocious", 10).unwrap(),
            vec!["supercalif", "ragilistic", "expialidoc", "ious"]
        );
        // The long-word chopper prefers a hyphen inside the window — but
        // here none lands in the first window, matching python3 exactly.
        assert_eq!(
            wrap("a supercalifragilistic-expialidocious word", 12).unwrap(),
            vec!["a supercalif", "ragilistic-e", "xpialidociou", "s word"]
        );
        assert_eq!(wrap("xxxxx", 70).unwrap(), vec!["xxxxx"]);
    }

    #[test]
    fn whitespace_munging_matches_python() {
        // Tabs expand column-aware (tabsize 8); newlines become spaces.
        assert_eq!(
            wrap("tabs\there\tand\nnewlines", 12).unwrap(),
            vec!["tabs    here", "and newlines"]
        );
        // The FIRST line keeps leading whitespace; later lines drop it.
        assert_eq!(
            wrap("  leading and   multiple   spaces  ", 10).unwrap(),
            vec!["  leading", "and", "multiple", "spaces"]
        );
        assert_eq!(wrap("", 10).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn invalid_width_raises_pythons_value_error() {
        let e = wrap("x", 0).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: invalid width 0 (must be > 0)");
        let e = fill("x", -3).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: invalid width -3 (must be > 0)");
    }
}

// ---------------------------------------------------------------------------
// hash() with PYTHONHASHSEED=0 semantics (issue #19). All values pinned
// against `PYTHONHASHSEED=0 python3`.
// ---------------------------------------------------------------------------

mod hash_builtin {
    use stdpython::*;

    #[test]
    fn int_and_bool_hashes_match_python() {
        assert_eq!(hash(&0i64), 0);
        assert_eq!(hash(&1i64), 1);
        // -1 is CPython's error marker: hash(-1) is -2.
        assert_eq!(hash(&-1i64), -2);
        assert_eq!(hash(&-2i64), -2);
        assert_eq!(hash(&((1i64 << 61) - 1)), 0);
        assert_eq!(hash(&(1i64 << 61)), 1);
        assert_eq!(hash(&true), 1);
        assert_eq!(hash(&false), 0);
    }

    #[test]
    fn string_hashes_match_pythons_zero_seed_siphash13() {
        assert_eq!(hash(""), 0);
        assert_eq!(hash("a"), 4644417185603328019);
        assert_eq!(hash("hello"), -2096571579003691106);
        assert_eq!(hash("café"), 137524001917817222);
        // UCS-2 and UCS-4 internal representations.
        assert_eq!(hash("日本"), 6243316497235261705);
        assert_eq!(hash("𝄞clef"), 456820485802690608);
    }

    #[test]
    fn float_hashes_match_python() {
        assert_eq!(hash(&1.5f64), 1152921504606846977);
        assert_eq!(hash(&0.5f64), 1152921504606846976);
        assert_eq!(hash(&2.0f64), 2);
        assert_eq!(hash(&-1.5f64), -1152921504606846977);
        assert_eq!(hash(&f64::INFINITY), 314159);
        assert_eq!(hash(&f64::NEG_INFINITY), -314159);
    }

    #[test]
    #[should_panic(expected = "hash(nan)")]
    fn nan_hash_fails_loudly() {
        let _ = hash(&f64::NAN);
    }
}

// ---------------------------------------------------------------------------
// csv.reader (issue #19). All expected values pinned against python3.
// ---------------------------------------------------------------------------

mod csv_module {
    use stdpython::csv::reader;

    fn rows(lines: &[&str]) -> Vec<Vec<String>> {
        reader(lines, b',', false, None).unwrap()
    }

    #[test]
    fn excel_dialect_parsing_matches_python() {
        assert_eq!(
            rows(&["a,b,c", "1,2,3"]),
            vec![vec!["a", "b", "c"], vec!["1", "2", "3"]]
        );
        // Quoted fields keep delimiters; "" escapes a quote.
        assert_eq!(rows(&["a,\"b,c\",d"]), vec![vec!["a", "b,c", "d"]]);
        assert_eq!(
            rows(&["a,\"say \"\"hi\"\"\",z"]),
            vec![vec!["a", "say \"hi\"", "z"]]
        );
        // Whitespace is data.
        assert_eq!(rows(&[" a , b "]), vec![vec![" a ", " b "]]);
        // Mid-field quotes are literal; data after a closing quote joins.
        assert_eq!(rows(&["a\"b,c"]), vec![vec!["a\"b", "c"]]);
        assert_eq!(rows(&["\"a\"b,c"]), vec![vec!["ab", "c"]]);
        assert_eq!(rows(&["\"a\"\"b\""]), vec![vec!["a\"b"]]);
    }

    #[test]
    fn empty_fields_lines_and_continuations_match_python() {
        // Empty fields, a bare comma, and an EMPTY line (an empty record).
        assert_eq!(
            rows(&["a,,c", ",", ""]),
            vec![
                vec!["a".to_string(), "".to_string(), "c".to_string()],
                vec!["".to_string(), "".to_string()],
                Vec::<String>::new()
            ]
        );
        // A quoted field spanning list elements joins them directly
        // (list elements carry no newline of their own).
        assert_eq!(rows(&["a,\"x", "y\",b"]), vec![vec!["a", "xy", "b"]]);
        // Unterminated quotes close at end of input, as in Python.
        assert_eq!(rows(&["a,\"unterminated"]), vec![vec!["a", "unterminated"]]);
        assert_eq!(rows(&["\"\",x"]), vec![vec!["", "x"]]);
    }

    #[test]
    fn readlines_style_newlines_terminate_records_like_python() {
        // readlines() keeps the line terminators; they end the record.
        assert_eq!(
            rows(&["a,b\n", "\n", "c\r\n", "d\r"]),
            vec![
                vec!["a".to_string(), "b".to_string()],
                Vec::<String>::new(),
                vec!["c".to_string()],
                vec!["d".to_string()]
            ]
        );
        // Inside quotes a newline is DATA — a quoted field spanning
        // readlines elements keeps it: python3 gives 'x\ny'.
        assert_eq!(rows(&["a,\"x\n", "y\"\n"]), vec![vec!["a", "x\ny"]]);
        // An unquoted newline with data after it is csv.Error.
        let e = reader(&["a\nb,c"], b',', false, None).unwrap_err();
        assert_eq!(
            format!("{}", e),
            "csv.Error: new-line character seen in unquoted field - do you \
             need to open the file with newline=''?"
        );
    }
}

mod csv_dialect_registry {
    use stdpython::stdlib::csv;

    #[test]
    fn delimiter_threads_through_reader_and_writer() {
        // A non-comma delimiter splits (semicolon; python3's
        // `list(csv.reader(['X;Y;Z'], delimiter=';'))`).
        let r = csv::reader(&["X;Y;Z"], b';', false, None).unwrap();
        assert_eq!(r[0], vec!["X", "Y", "Z"]);
        // A tab delimiter.
        let r = csv::reader(&["a\tb"], b'\t', false, None).unwrap();
        assert_eq!(r[0], vec!["a", "b"]);
    }

    #[test]
    fn register_get_unregister_list_round_trip() {
        // Default registry is CPython's built-in dialects.
        assert!(csv::list_dialects().contains(&"excel".to_string()));
        assert!(csv::list_dialects().contains(&"excel-tab".to_string()));
        // register_dialect(name, delimiter=...) then the delimiter resolves.
        csv::register_dialect("semi_reg", Some(b';'), None, None, None, None, None, None, None)
            .unwrap();
        assert_eq!(csv::dialect_delimiter("semi_reg").unwrap(), b';');
        // An unknown dialect name is csv.Error, not a silent default
        // (verified against python3: `csv.reader(['a;b'], 'nonesuch')`
        // raises `csv.Error: unknown dialect`).
        let e = csv::dialect_delimiter("nonesuch").unwrap_err();
        assert_eq!(format!("{}", e), "csv.Error: unknown dialect");
        let d = csv::get_dialect("semi_reg").unwrap();
        assert_eq!(d.delimiter, b';');
        assert!(csv::get_dialect("nonesuch").is_err());
        csv::unregister_dialect("semi_reg").unwrap();
        assert!(csv::get_dialect("semi_reg").is_err());
        // The built-in unix dialect is QUOTE_ALL (verified against python3).
        assert_eq!(csv::get_dialect("unix").unwrap().quoting, csv::QUOTE_ALL);
    }

    #[test]
    fn named_dialect_drives_the_reader_like_python() {
        // python3: csv.register_dialect('semi', delimiter=';');
        // list(csv.reader(['X;Y;Z'], 'semi')) == [['X','Y','Z']]
        csv::register_dialect("semi_named", Some(b';'), None, None, None, None, None, None, None)
            .unwrap();
        let delim = csv::dialect_delimiter("semi_named").unwrap();
        let r = csv::reader(&["X;Y;Z"], delim, false, None).unwrap();
        assert_eq!(r[0], vec!["X", "Y", "Z"]);
        csv::unregister_dialect("semi_named").unwrap();
    }
}

mod print_display {
    use stdpython::py_display;

    #[test]
    fn py_display_matches_python_str() {
        // Python's str(): True/False, exponent-form large floats,
        // unquoted strings — none of which Rust's Display produces.
        assert_eq!(py_display(&true), "True");
        assert_eq!(py_display(&false), "False");
        assert_eq!(py_display(&42i64), "42");
        assert_eq!(py_display(&1e16f64), "1e+16");
        assert_eq!(py_display(&2.5f64), "2.5");
        assert_eq!(py_display(&"plain"), "plain");
        assert_eq!(py_display(&String::from("owned")), "owned");
        // len() yields usize; it prints like any int.
        assert_eq!(py_display(&3usize), "3");
    }

    #[test]
    fn containers_use_repr_for_elements_and_none_is_none() {
        // str(['a']) is "['a']" — element strings keep their quotes.
        assert_eq!(py_display(&vec![1i64, 2, 3]), "[1, 2, 3]");
        assert_eq!(
            py_display(&vec!["a".to_string(), "b".to_string()]),
            "['a', 'b']"
        );
        // Option-based None model: None prints as None, a present string
        // prints unquoted (str, not repr).
        assert_eq!(py_display(&Option::<i64>::None), "None");
        assert_eq!(py_display(&Some("hi")), "hi");
        assert_eq!(py_display(&Some(2.5f64)), "2.5");
    }
}

mod list_sort {
    use stdpython::PySort;

    #[test]
    fn py_sort_shapes_match_python() {
        let mut xs = vec![3i64, 1, 2];
        xs.py_sort();
        assert_eq!(xs, vec![1, 2, 3]);
        xs.py_sort_reverse(true);
        assert_eq!(xs, vec![3, 2, 1]);
        xs.py_sort_reverse(false);
        assert_eq!(xs, vec![1, 2, 3]);

        // Floats sort (Vec's inherent sort would reject them).
        let mut ys = vec![2.5f64, -1.0, 0.5];
        ys.py_sort();
        assert_eq!(ys, vec![-1.0, 0.5, 2.5]);

        // key= runs once per element; lengths fig(3) < pear(4) <
        // banana(6) order the words.
        let mut words = vec!["pear".to_string(), "fig".to_string(), "banana".to_string()];
        words.py_sort_key(|w| w.chars().count() as i64);
        assert_eq!(words, vec!["fig", "pear", "banana"]);
        words.py_sort_key_reverse(|w| w.chars().count() as i64, true);
        assert_eq!(words, vec!["banana", "pear", "fig"]);
    }

    #[test]
    fn reverse_true_is_stable_not_a_reversal() {
        // Python's reverse=True keeps EQUAL keys in source order; a
        // sort-then-reverse would flip them. python3:
        // sorted([(1,'a'),(2,'x'),(1,'b')], key=lambda t:t[0], reverse=True)
        // == [(2,'x'),(1,'a'),(1,'b')]
        let mut pairs = vec![
            (1i64, "a".to_string()),
            (2i64, "x".to_string()),
            (1i64, "b".to_string()),
        ];
        pairs.py_sort_key_reverse(|t| t.0, true);
        assert_eq!(
            pairs,
            vec![
                (2i64, "x".to_string()),
                (1i64, "a".to_string()),
                (1i64, "b".to_string())
            ]
        );
    }

    #[test]
    fn nan_sort_panics_loudly() {
        let mut xs = vec![1.0f64, f64::NAN];
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            xs.py_sort();
        }))
        .unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(msg.contains("NaN"), "panic message: {}", msg);
    }
}

mod re_named_groups {
    use stdpython::re;
    use stdpython::{PyMatchOps, PyRepr};

    #[test]
    fn group_name_and_groupdict_match_python() {
        let m = re::search(r"(?P<user>\w+)@(?P<host>[\w.]+)", "bob@example.com", "").unwrap();
        assert_eq!(m.group_name("user"), "bob");
        assert_eq!(m.group_name("host"), "example.com");
        // Numeric access still works alongside names.
        assert_eq!(m.group(1), "bob");
        let d = m.groupdict();
        assert_eq!(d.get("user").map(String::as_str), Some("bob"));
        assert_eq!(d.get("host").map(String::as_str), Some("example.com"));
        // Insertion follows group-index order, like Python's dict.
        let keys: Vec<&String> = d.keys().collect();
        assert_eq!(keys, ["user", "host"]);
    }

    #[test]
    fn unknown_group_name_is_index_error() {
        let m = re::search(r"(?P<x>a)", "a", "").unwrap();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.group_name("nope");
        }))
        .unwrap_err();
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
        // Python: IndexError: no such group
        assert!(msg.contains("IndexError"), "panic message: {}", msg);
        assert!(msg.contains("no such group"), "panic message: {}", msg);
    }

    #[test]
    fn findall_tuple_variants_match_python() {
        // python3: re.findall(r"(\w+)=(\d+)", "a=1 b=22") == [('a','1'),('b','22')]
        assert_eq!(
            re::findall2(r"(\w+)=(\d+)", "a=1 b=22", "").unwrap(),
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "22".to_string())
            ]
        );
        // A non-participating group contributes '' in findall, as Python.
        assert_eq!(
            re::findall2(r"(a)|(b)", "ab", "").unwrap(),
            vec![
                ("a".to_string(), "".to_string()),
                ("".to_string(), "b".to_string())
            ]
        );
        assert_eq!(
            re::findall3(r"(\d+)-(\d+)-(\d+)", "2024-01-05", "").unwrap(),
            vec![("2024".to_string(), "01".to_string(), "05".to_string())]
        );
        // The tuple arity is part of the TYPE: a mismatched pattern is a
        // loud error, never a mis-shaped result.
        let err = re::findall2(r"(\d+)", "1", "").unwrap_err();
        assert!(format!("{}", err).contains("capture groups"), "{}", err);
    }

    #[test]
    fn tuples_repr_like_python() {
        // python3: repr(('a', '1')) == "('a', '1')"
        assert_eq!(("a".to_string(), "1".to_string()).py_repr(), "('a', '1')");
        assert_eq!((1i64, 2i64, 3i64).py_repr(), "(1, 2, 3)");
        // Through Vec: [('a', '1')]
        assert_eq!(
            vec![("a".to_string(), "1".to_string())].py_repr(),
            "[('a', '1')]"
        );
    }
}

mod datetime_fields_and_directives {
    use stdpython::datetime::datetime;

    #[test]
    fn datetime_fields_are_flat_like_python() {
        let d = datetime::new(2024, 2, 29, Some(13), Some(5), Some(7), Some(123456)).unwrap();
        assert_eq!(
            (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond),
            (2024, 2, 29, 13, 5, 7, 123456)
        );
        // dt.date() / dt.time() are methods, as in Python.
        assert_eq!(format!("{}", d.date()), "2024-02-29");
        assert_eq!(format!("{}", d.time()), "13:05:07.123456");
    }

    #[test]
    fn strptime_julian_day_matches_python() {
        // python3: strptime("2024-060", "%Y-%j") == 2024-02-29
        let d = datetime::strptime("2024-060", "%Y-%j").unwrap();
        assert_eq!((d.year, d.month, d.day), (2024, 2, 29));
        // Day 366 of a 365-day year rolls into the next year (ordinal
        // arithmetic, as CPython).
        let d = datetime::strptime("2023-366", "%Y-%j").unwrap();
        assert_eq!((d.year, d.month, d.day), (2024, 1, 1));
        // Without a year, 1900: python3 gives 1900-03-01 for day 60.
        let d = datetime::strptime("060", "%j").unwrap();
        assert_eq!((d.year, d.month, d.day), (1900, 3, 1));
        // CPython's %j matches the LONGEST in-range digit prefix: "367"
        // consumes "36" and the trailing "7" is unconverted data.
        let e = datetime::strptime("2023-367", "%Y-%j").unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: unconverted data remains: 7");
        // 000 matches nothing.
        let e = datetime::strptime("2023-000", "%Y-%j").unwrap_err();
        assert!(format!("{}", e).contains("does not match format"), "{}", e);
    }

    #[test]
    fn strptime_weekday_names_parse_and_are_ignored() {
        // 2024-01-02 is a Tuesday; CPython does NOT validate the parsed
        // weekday against the date.
        let d = datetime::strptime("Mon 2024-01-02", "%a %Y-%m-%d").unwrap();
        assert_eq!((d.year, d.month, d.day), (2024, 1, 2));
        // Full names via %A, case-insensitively.
        let d = datetime::strptime("friday 2024-03-01", "%A %Y-%m-%d").unwrap();
        assert_eq!((d.year, d.month, d.day), (2024, 3, 1));
        // Combined with %j.
        let d = datetime::strptime("Tue 2024-060", "%a %Y-%j").unwrap();
        assert_eq!((d.year, d.month, d.day), (2024, 2, 29));
        // A non-weekday is Python's mismatch ValueError.
        let e = datetime::strptime("Xyz 2024-01-02", "%a %Y-%m-%d").unwrap_err();
        assert_eq!(
            format!("{}", e),
            "ValueError: time data 'Xyz 2024-01-02' does not match format '%a %Y-%m-%d'"
        );
        // %a takes only abbreviated names: "Monday" consumes "Mon" and
        // the leftover "day" breaks the format, as in CPython.
        let e = datetime::strptime("Monday 2024-01-02", "%a %Y-%m-%d").unwrap_err();
        assert_eq!(
            format!("{}", e),
            "ValueError: time data 'Monday 2024-01-02' does not match format '%a %Y-%m-%d'"
        );
    }
}

mod replace_keywords {
    use stdpython::datetime::{date, datetime, time, PyReplace, ReplaceArgs};

    #[test]
    fn replace_maps_fields_per_receiver_type() {
        let d = datetime::new(2024, 2, 29, Some(13), Some(5), Some(7), Some(123456)).unwrap();
        let r = d
            .py_replace(ReplaceArgs {
                hour: Some(14),
                ..ReplaceArgs::default()
            })
            .unwrap();
        assert_eq!(format!("{}", r), "2024-02-29 14:05:07.123456");

        let dd = date::new(2024, 2, 29).unwrap();
        let r = dd
            .py_replace(ReplaceArgs {
                month: Some(3),
                day: Some(1),
                ..ReplaceArgs::default()
            })
            .unwrap();
        assert_eq!(format!("{}", r), "2024-03-01");

        let t = time::new(13, 5, Some(7), Some(0)).unwrap();
        let r = t
            .py_replace(ReplaceArgs {
                minute: Some(0),
                ..ReplaceArgs::default()
            })
            .unwrap();
        assert_eq!(format!("{}", r), "13:00:07");
    }

    #[test]
    fn foreign_fields_raise_pythons_type_error() {
        // CPython: TypeError: 'hour' is an invalid keyword argument for replace()
        let dd = date::new(2024, 2, 29).unwrap();
        let e = dd
            .py_replace(ReplaceArgs {
                hour: Some(1),
                ..ReplaceArgs::default()
            })
            .unwrap_err();
        assert_eq!(
            format!("{}", e),
            "TypeError: 'hour' is an invalid keyword argument for replace()"
        );

        let t = time::new(1, 2, None, None).unwrap();
        let e = t
            .py_replace(ReplaceArgs {
                year: Some(2000),
                ..ReplaceArgs::default()
            })
            .unwrap_err();
        assert_eq!(
            format!("{}", e),
            "TypeError: 'year' is an invalid keyword argument for replace()"
        );
    }

    #[test]
    fn out_of_range_values_raise_value_error() {
        let d = datetime::new(2024, 1, 15, None, None, None, None).unwrap();
        // python3: d.replace(month=2, day=30) -> ValueError: day is out of
        // range for month
        assert!(d
            .py_replace(ReplaceArgs {
                month: Some(2),
                day: Some(30),
                ..ReplaceArgs::default()
            })
            .is_err());
        // A negative field cannot narrow to u32: ValueError, not a wrap.
        assert!(d
            .py_replace(ReplaceArgs {
                hour: Some(-1),
                ..ReplaceArgs::default()
            })
            .is_err());
    }
}

mod file_objects {
    use stdpython::{csv, io};

    #[test]
    fn stringio_cursor_semantics_match_python() {
        // python3: StringIO("seeded").write("!") OVERWRITES at the
        // cursor: buffer becomes "!eeded", cursor 1, read() -> "eeded".
        let b = io::StringIO_seeded("seeded");
        assert_eq!(b.write("!").unwrap(), 1);
        assert_eq!(b.getvalue().unwrap(), "!eeded");
        assert_eq!(b.read().unwrap(), "eeded");
        // At the end, write appends; write returns the CHAR count.
        assert_eq!(b.write("après").unwrap(), 5);
        assert_eq!(b.getvalue().unwrap(), "!eededaprès");

        // readline/readlines keep terminators, as Python.
        let two = io::StringIO_seeded("x\ny\nz");
        assert_eq!(two.readline().unwrap(), "x\n");
        assert_eq!(two.readlines().unwrap(), vec!["y\n", "z"]);
        // Exhausted: empty line, empty list.
        assert_eq!(two.readline().unwrap(), "");
    }

    #[test]
    fn file_line_iteration_matches_python() {
        // list(io.StringIO("x\ny\n\nz")) -> ['x\n', 'y\n', '\n', 'z'],
        // and the file is exhausted after: f.readline() -> ''
        let f = io::StringIO_seeded("x\ny\n\nz");
        let lines: Vec<String> = f.py_lines().map(Result::unwrap).collect();
        assert_eq!(lines, vec!["x\n", "y\n", "\n", "z"]);
        assert_eq!(f.readline().unwrap(), "");
        // for l in f: break  -> 'a\n'; then list(f) -> ['b\n'] (the
        // loop and the later reads share one cursor)
        let f = io::StringIO_seeded("a\nb\n");
        assert_eq!(f.py_lines().next().unwrap().unwrap(), "a\n");
        let rest: Vec<String> = f.py_lines().map(Result::unwrap).collect();
        assert_eq!(rest, vec!["b\n"]);
        // f.close(); for l in f -> ValueError: I/O operation on closed
        // file. — raised once, ending the iteration
        f.close().unwrap();
        let mut closed = f.py_lines();
        let err = closed.next().unwrap().unwrap_err();
        assert_eq!(err.type_name(), "ValueError");
        assert_eq!(err.message, "I/O operation on closed file.");
        assert!(closed.next().is_none());
    }

    #[test]
    fn closed_files_raise_pythons_value_error() {
        let b = io::StringIO();
        b.close().unwrap();
        let e = b.read().unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: I/O operation on closed file.");
        // getvalue on a closed buffer is closed too.
        assert!(b.getvalue().is_err());
    }

    #[test]
    fn csv_writer_quoting_matches_python() {
        // python3 (excel dialect):
        // 'a,"b,c","say ""hi""",\r\n1,2,3\r\n\r\n"line\nbreak",tab\there\r\n'
        let mut buf = io::StringIO();
        {
            let mut w = csv::writer(&mut buf, "\r\n".to_string(), b',', false, None);
            w.writerow(&["a", "b,c", "say \"hi\"", ""]).unwrap();
            w.writerow(&[1i64, 2, 3]).unwrap();
            w.writerow(&[] as &[&str]).unwrap();
            w.writerow(&["line\nbreak", "tab\there"]).unwrap();
        }
        assert_eq!(
            buf.getvalue().unwrap(),
            "a,\"b,c\",\"say \"\"hi\"\"\",\r\n1,2,3\r\n\r\n\"line\nbreak\",tab\there\r\n"
        );

        // Elements stringify through PyDisplay (Python's str()): bools
        // and floats render as Python prints them.
        let mut buf = io::StringIO();
        {
            let mut w = csv::writer(&mut buf, "\r\n".to_string(), b',', false, None);
            w.writerow(&[stdpython::py_display(&true), stdpython::py_display(&2.5f64)])
                .unwrap();
            w.writerows(&[vec!["x", "y"], vec!["z", "w"]]).unwrap();
        }
        assert_eq!(buf.getvalue().unwrap(), "True,2.5\r\nx,y\r\nz,w\r\n");

        // writer output round-trips through the reader.
        let mut buf = io::StringIO();
        {
            let mut w = csv::writer(&mut buf, "\r\n".to_string(), b',', false, None);
            w.writerow(&["a", "b,c", "say \"hi\""]).unwrap();
        }
        let text = buf.getvalue().unwrap();
        let rows = csv::reader(&text.split("\r\n").collect::<Vec<_>>(), b',', false, None).unwrap();
        assert_eq!(rows[0], vec!["a", "b,c", "say \"hi\""]);
    }
}

mod lru_cache_store {
    use stdpython::PyLruCache;

    #[test]
    fn hits_touch_and_eviction_drops_least_recent() {
        let mut c: PyLruCache<(i64,), i64> = PyLruCache::new(Some(2));
        c.put((1,), 10);
        c.put((2,), 20);
        // Touch 1 so 2 becomes least-recently-used...
        assert_eq!(c.get(&(1,)), Some(10));
        // ...then inserting 3 evicts 2, not 1 (CPython's discipline).
        c.put((3,), 30);
        assert_eq!(c.get(&(2,)), None);
        assert_eq!(c.get(&(1,)), Some(10));
        assert_eq!(c.get(&(3,)), Some(30));
    }

    #[test]
    fn unbounded_cache_never_evicts() {
        let mut c: PyLruCache<(i64,), i64> = PyLruCache::new(None);
        for i in 0..1000 {
            c.put((i,), i * 2);
        }
        assert_eq!(c.get(&(0,)), Some(0));
        assert_eq!(c.get(&(999,)), Some(1998));
    }
}

mod dict_and_exception_display {
    use stdpython::{py_display, PyDict, PyException, PyRepr};

    #[test]
    fn dict_repr_matches_python_including_order() {
        // python3: repr({'a': 1, 'b': 'x'}) == "{'a': 1, 'b': 'x'}" —
        // keys AND values use repr, and insertion order is preserved.
        let mut d: PyDict<String, i64> = PyDict::default();
        d.insert("b".to_string(), 2);
        d.insert("a".to_string(), 1);
        assert_eq!(d.py_repr(), "{'b': 2, 'a': 1}");
        assert_eq!(py_display(&d), "{'b': 2, 'a': 1}");

        let empty: PyDict<String, i64> = PyDict::default();
        assert_eq!(empty.py_repr(), "{}");

        // len() yields usize, so a container of lengths must still
        // render — PyRepr covers every integer width, as PyDisplay does.
        let mut lens: PyDict<String, usize> = PyDict::default();
        lens.insert("ab".to_string(), 2usize);
        assert_eq!(lens.py_repr(), "{'ab': 2}");
        assert_eq!(vec![1usize, 2].py_repr(), "[1, 2]");

        // Values render with repr: strings keep their quotes, floats and
        // bools use Python's spelling.
        let mut mixed: PyDict<String, String> = PyDict::default();
        mixed.insert("k".to_string(), "v".to_string());
        assert_eq!(mixed.py_repr(), "{'k': 'v'}");
    }

    #[test]
    fn exception_str_is_the_message_alone() {
        // python3: str(ValueError("boom")) == "boom" — NOT the
        // "ValueError: boom" traceback form that Display produces.
        let e = PyException::new("ValueError", "boom");
        assert_eq!(py_display(&e), "boom");
        assert_eq!(format!("{}", e), "ValueError: boom");
        // python3: repr(ValueError("boom")) == "ValueError('boom')"
        assert_eq!(e.py_repr(), "ValueError('boom')");
    }
}

mod cpython_numeric_and_stdlib_fixes {
    use stdpython::{datetime::date, math, py_pow, py_str_repr, PyFloat, PyInt, PyMul};

    #[test]
    fn float_power_uses_libm_pow_not_repeated_squaring() {
        // python3: 0.1 ** 4 == 0.00010000000000000002 (powi's repeated
        // squaring gives ...05), and 1.05 ** 10 == 1.628894626777442.
        assert_eq!(py_pow(0.1f64, 4i64), 0.1f64.powf(4.0));
        assert_eq!(format!("{:?}", py_pow(0.1f64, 4i64)), "0.00010000000000000002");
        assert_eq!(format!("{:?}", py_pow(1.05f64, 10i64)), "1.628894626777442");
    }

    #[test]
    fn numeric_strings_follow_pythons_underscore_grammar() {
        // Round 10 of the review on #339: `int(s)` and `float(s)` accept
        // single underscores BETWEEN digits only — `_1`, `1_` and `1__2`
        // are CPython's ValueError, never a number with the underscores
        // dropped. Every row is python3 3.11's answer (VE = ValueError).
        let table: &[(&str, Option<i64>, Option<f64>)] = &[
            ("_1", None, None),
            ("1_", None, None),
            ("1__2", None, None),
            ("1_000", Some(1000), Some(1000.0)),
            (" 1_0 ", Some(10), Some(10.0)),
            ("+1_0", Some(10), Some(10.0)),
            ("-1_0", Some(-10), Some(-10.0)),
            ("1_0_0", Some(100), Some(100.0)),
            ("0_1", Some(1), Some(1.0)),
            ("1 0", None, None),
            ("", None, None),
            ("  ", None, None),
            ("1_0.5", None, Some(10.5)),
            ("1_.0", None, None),
            ("1__0.5", None, None),
            ("1e1_0", None, Some(1e10)),
            ("1_e10", None, None),
            ("1.5_", None, None),
            ("_1.5", None, None),
            (".5", None, Some(0.5)),
            (".5_1", None, Some(0.51)),
            ("5.", None, Some(5.0)),
            ("5._1", None, None),
            ("1e_1", None, None),
            ("in_f", None, None),
            ("inf", None, Some(f64::INFINITY)),
            ("-Infinity", None, Some(f64::NEG_INFINITY)),
            ("1_0e-1_0", None, Some(1e-9)),
            ("1.", None, Some(1.0)),
            ("-.5e+2", None, Some(-50.0)),
            ("1e", None, None),
            ("e1", None, None),
            ("0x1_0", None, None),
            ("1_0j", None, None),
            // i64::MIN, whose magnitude is not an i64, reads as CPython
            // reads it (round 11).
            ("-9223372036854775808", Some(i64::MIN), Some(-9223372036854775808.0)),
            ("9223372036854775807", Some(i64::MAX), Some(9223372036854775807.0)),
        ];
        for (text, int, float) in table {
            match (text.py_int(), int) {
                (Ok(v), Some(want)) => assert_eq!(v, *want, "int({:?})", text),
                (Err(e), None) => assert_eq!(
                    e.to_string(),
                    format!("ValueError: invalid literal for int() with base 10: {}", py_str_repr(text)),
                    "int({:?})",
                    text
                ),
                (got, want) => panic!("int({:?}): got {:?}, python3 gives {:?}", text, got, want),
            }
            match (text.py_float(), float) {
                (Ok(v), Some(want)) => assert_eq!(v, *want, "float({:?})", text),
                (Err(e), None) => assert_eq!(
                    e.to_string(),
                    format!("ValueError: could not convert string to float: {}", py_str_repr(text)),
                    "float({:?})",
                    text
                ),
                (got, want) => panic!("float({:?}): got {:?}, python3 gives {:?}", text, got, want),
            }
        }
        // python3: float("nan") is nan (both spellings), and the message
        // quotes the string as Python's repr does.
        assert!("+nan".py_float().unwrap().is_nan());
        assert!("NaN".py_float().unwrap().is_nan());
        // python3: math.copysign(1.0, float("-nan")) == -1.0 — the sign
        // survives on a nan (round 11); "+nan" and "nan" are positive.
        assert!("-nan".py_float().unwrap().is_sign_negative());
        assert!(" -NaN ".py_float().unwrap().is_sign_negative());
        assert!("+nan".py_float().unwrap().is_sign_positive());
        assert!("nan".py_float().unwrap().is_sign_positive());
        assert_eq!(math::copysign(1.0, "-nan".py_float().unwrap()), -1.0);
        assert_eq!(
            "it's".py_int().unwrap_err().to_string(),
            "ValueError: invalid literal for int() with base 10: \"it's\""
        );
        // What CPython reads and the runtime's i64 cannot hold is loud,
        // never a different value: non-ASCII decimal digits (python3:
        // int("٣") == 3) and an int outside i64.
        let e = "٣".py_int().unwrap_err();
        assert_eq!(e.to_string(), "NotImplementedError: int('٣'): non-ASCII digits are not supported yet");
        let e = "99999999999999999999".py_int().unwrap_err();
        assert_eq!(
            e.to_string(),
            "NotImplementedError: int('99999999999999999999'): an int outside i64 are not supported yet"
        );
    }

    #[test]
    fn int_conversions_match_python() {
        // Python strips surrounding whitespace and allows _ separators,
        // so int(line) over file lines works.
        assert_eq!("42\n".py_int().unwrap(), 42);
        assert_eq!(" 7 ".py_int().unwrap(), 7);
        assert_eq!("1_000".py_int().unwrap(), 1000);
        // python3: int(b"\xff"[0]) == 255 — a bytes element is already an
        // int, so int() over one is the identity.
        assert_eq!(0xffu8.py_int().unwrap(), 255);
        // python3: "ab" * 3 == "ababab"; "x" * 0 == ""; "y" * -2 == "";
        // 3 * "ab" == "abab" * 1 — repetition in either operand order.
        assert_eq!("ab".to_string().py_mul(&3), "ababab");
        assert_eq!("x".py_mul(&0), "");
        assert_eq!("y".to_string().py_mul(&-2), "");
        assert_eq!(2i64.py_mul(&"ab".to_string()), "abab");
        // NaN and infinity raise instead of silently becoming 0/i64::MAX.
        let e = f64::NAN.py_int().unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: cannot convert float NaN to integer");
        let e = f64::INFINITY.py_int().unwrap_err();
        assert_eq!(
            format!("{}", e),
            "OverflowError: cannot convert float infinity to integer"
        );
    }

    #[test]
    fn isocalendar_follows_iso_week_rules() {
        // python3: these dates belong to the NEIGHBOURING ISO year.
        assert_eq!(date::new(2023, 1, 1).unwrap().isocalendar(), (2022, 52, 7));
        assert_eq!(date::new(2024, 12, 31).unwrap().isocalendar(), (2025, 1, 2));
        assert_eq!(date::new(2024, 12, 30).unwrap().isocalendar(), (2025, 1, 1));
        // And an ordinary mid-year date.
        assert_eq!(date::new(2000, 3, 1).unwrap().isocalendar(), (2000, 9, 3));
        assert_eq!(date::new(2026, 7, 27).unwrap().isocalendar(), (2026, 31, 1));
    }

    #[test]
    fn deque_maxlen_discards_from_the_opposite_end() {
        use stdpython::collections::deque;
        // python3: deque([1,2,3], maxlen=3).appendleft(0) -> deque([0,1,2])
        // — always popping the front discarded the element just added.
        let mut d: deque<i64> = deque::with_maxlen(3);
        d.extend(vec![1, 2, 3]);
        d.appendleft(0);
        assert_eq!(
            (0..3).map(|i| *d.get(i).unwrap()).collect::<Vec<i64>>(),
            vec![0, 1, 2]
        );
        // Growing at the back still evicts the front.
        let mut d: deque<i64> = deque::with_maxlen(3);
        d.extend(vec![1, 2, 3]);
        d.append(4);
        assert_eq!(
            (0..3).map(|i| *d.get(i).unwrap()).collect::<Vec<i64>>(),
            vec![2, 3, 4]
        );
    }
}

#[test]
fn os_path_expandvars_matches_python() {
    // Verified against python3:
    //   os.environ['RY_TEST_V']='hello'
    //   os.path.expandvars('$RY_TEST_V/x')   -> 'hello/x'
    //   os.path.expandvars('${RY_TEST_V}-t') -> 'hello-t'
    //   os.path.expandvars('no$UNSET var')   -> 'no$UNSET var'
    //   os.path.expandvars('plain')          -> 'plain'
    unsafe {
        std::env::set_var("RY_TEST_V", "hello");
    }
    use stdpython::stdlib::os::path::expandvars;
    assert_eq!(expandvars("$RY_TEST_V/x"), "hello/x");
    assert_eq!(expandvars("${RY_TEST_V}-tail"), "hello-tail");
    // Unknown variables stay literal — bare AND braced forms
    // (CPython: os.path.expandvars('${RY_UNSET_X}/log') keeps the text).
    assert_eq!(expandvars("no$RY_UNSET_X var"), "no$RY_UNSET_X var");
    assert_eq!(expandvars("${RY_UNSET_X}/log"), "${RY_UNSET_X}/log");
    assert_eq!(expandvars("a${RY_UNSET_X}b"), "a${RY_UNSET_X}b");
    // CPython's varscan is ASCII-only (`re.compile(r'\$(\w+|\{[^}]*\})',
    // re.ASCII)`): the bare name is [A-Za-z0-9_]+ with NO first-character
    // rule. A Unicode letter ends the scan (`$naive` scans `na`, unset,
    // so the text stays), and a DIGIT-LEADING name behaves like any
    // other: expanded when the variable exists, literal when not.
    // Verified against python3 3.14:
    //   1abc=hello python3 -c "...expandvars('$1abc')..." -> 'hello'
    //   ...expandvars('${1abc}-x')...                     -> 'hello-x'
    //   ...expandvars('$9zzz') (unset)                    -> '$9zzz'
    unsafe {
        std::env::set_var("1abc", "digit-led");
    }
    assert_eq!(expandvars("$naive"), "$naive");
    assert_eq!(expandvars("$1abc"), "digit-led");
    assert_eq!(expandvars("${1abc}-x"), "digit-led-x");
    assert_eq!(expandvars("$9zzz"), "$9zzz");
    // The whole `${...}` span is ONE varscan token: when the variable is
    // unset, an inner `$b` must NOT be re-expanded even though `b` is
    // set — CPython keeps the literal text and advances past the brace.
    // Verified against python3 3.14 (with b=XX):
    //   ...expandvars('${a$b}')        -> '${a$b}'
    //   ...expandvars('pre${a$b}post') -> 'pre${a$b}post'
    //   ...expandvars('${}')           -> '${}'
    //   ...expandvars('${a{b}}')       -> '${a{b}}'
    unsafe {
        std::env::set_var("RY_TEST_B", "XX");
    }
    assert_eq!(expandvars("${a$RY_TEST_B}"), "${a$RY_TEST_B}");
    assert_eq!(expandvars("pre${a$RY_TEST_B}post"), "pre${a$RY_TEST_B}post");
    assert_eq!(expandvars("${}"), "${}");
    assert_eq!(expandvars("${a{RY_TEST_B}}}"), "${a{RY_TEST_B}}}");
    // An UNTERMINATED `${...` never matches the braced token, so the
    // scan falls through and a later bare reference still expands,
    // exactly like CPython's regex failing at that position.
    // Verified against python3 3.14:
    //   ...expandvars('x${abc$RY_TEST_B') -> 'x${abcXX'
    assert_eq!(expandvars("x${abc$RY_TEST_B"), "x${abcXX");
    assert_eq!(expandvars("plain"), "plain");
    assert_eq!(expandvars("$RY_TEST_V$RY_TEST_V"), "hellohello");
}

mod threading_module {
    use stdpython::threading;

    #[test]
    fn thread_lifecycle_matches_python() {
        // Verified against python3: is_alive() is False before start and
        // after join; join() waits for the body.
        let (tx, rx) = std::sync::mpsc::channel::<i64>();
        let t = threading::Thread::new("worker", false, move || {
            tx.send(42).unwrap();
        });
        assert!(!t.is_alive());
        t.start();
        t.join();
        assert!(!t.is_alive());
        assert_eq!(rx.recv().unwrap(), 42);
    }

    #[test]
    #[should_panic(expected = "threads can only be started once")]
    fn double_start_raises_pythons_runtime_error() {
        // Verified against python3: RuntimeError('threads can only be started once')
        let t = threading::Thread::new("worker", false, || {});
        t.start();
        t.join();
        t.start();
    }

    #[test]
    #[should_panic(expected = "cannot join thread before it is started")]
    fn join_before_start_raises_pythons_runtime_error() {
        // Verified against python3: RuntimeError('cannot join thread before it is started')
        let t = threading::Thread::new("worker", false, || {});
        t.join();
    }

    #[test]
    fn lock_matches_python() {
        // Verified against python3: acquire() -> True, locked() flips,
        // release() of an unlocked lock -> RuntimeError('release unlocked lock').
        let lock = threading::Lock();
        assert!(!lock.locked());
        assert!(lock.acquire().unwrap());
        assert!(lock.locked());
        lock.release().unwrap();
        assert!(!lock.locked());
        let e = lock.release().unwrap_err();
        assert_eq!(format!("{}", e), "RuntimeError: release unlocked lock");
        // The with-statement guard releases on drop.
        {
            let _g = lock.py_guard().unwrap();
            assert!(lock.locked());
        }
        assert!(!lock.locked());
    }

    #[test]
    fn lock_excludes_across_threads() {
        // Two threads bump a shared counter under the lock; the final
        // value proves every increment was mutually excluded.
        let lock = threading::Lock();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let lock = lock.clone();
            let counter = counter.clone();
            let t = threading::Thread::new("bump", false, move || {
                for _ in 0..100 {
                    let _g = lock.py_guard().unwrap();
                    let v = counter.load(std::sync::atomic::Ordering::SeqCst);
                    counter.store(v + 1, std::sync::atomic::Ordering::SeqCst);
                }
            });
            t.start();
            handles.push(t);
        }
        for t in &handles {
            t.join();
        }
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 400);
    }

    #[test]
    fn rlock_is_reentrant_and_owner_checked() {
        // Verified against python3: nested acquire on one thread works;
        // release without acquire -> RuntimeError('cannot release un-acquired lock').
        let rl = threading::RLock();
        assert!(rl.acquire().unwrap());
        assert!(rl.acquire().unwrap());
        rl.release().unwrap();
        rl.release().unwrap();
        let e = rl.release().unwrap_err();
        assert_eq!(
            format!("{}", e),
            "RuntimeError: cannot release un-acquired lock"
        );
    }

    #[test]
    fn event_signals_a_waiting_thread() {
        // Verified against python3: is_set() False -> set() -> wait() True.
        let ev = threading::Event();
        assert!(!ev.is_set());
        let ev2 = ev.clone();
        let t = threading::Thread::new("setter", false, move || {
            ev2.set();
        });
        t.start();
        assert!(ev.wait().unwrap());
        t.join();
        assert!(ev.is_set());
        let mut ev = ev;
        ev.clear();
        assert!(!ev.is_set());
    }

    #[test]
    fn semaphore_counts_and_blocks_at_zero() {
        // Verified against python3: acquire() -> True; a release unblocks
        // a waiter.
        let sem = threading::Semaphore(1);
        assert!(sem.acquire().unwrap());
        let sem2 = sem.clone();
        let t = threading::Thread::new("waiter", false, move || {
            // Blocks until the main thread releases.
            sem2.acquire().unwrap();
            sem2.release().unwrap();
        });
        t.start();
        sem.release().unwrap();
        t.join();
    }

    #[test]
    #[should_panic(expected = "semaphore initial value must be >= 0")]
    fn negative_semaphore_raises_pythons_value_error() {
        // Verified against python3: ValueError('semaphore initial value must be >= 0')
        threading::Semaphore(-1);
    }

    #[test]
    fn current_thread_and_active_count() {
        // Verified against python3: spawned threads are named
        // 'Thread-N (target)'; active_count() counts the main thread.
        // (The 'MainThread' name is pinned by the end-to-end convert test
        // — the cargo test harness runs tests on NAMED worker threads, so
        // the main-thread mapping is not observable here.)
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let t = threading::Thread::new("namer", false, move || {
            tx.send(threading::current_thread().name).unwrap();
        });
        t.start();
        t.join();
        let name = rx.recv().unwrap();
        assert!(
            name.starts_with("Thread-") && name.ends_with(" (namer)"),
            "CPython thread naming: {}",
            name
        );
        assert!(threading::active_count() >= 1);
    }
}

mod socket_module {
    use stdpython::{socket, threading};

    #[test]
    fn tcp_echo_roundtrip() {
        // A loopback echo: server thread accepts one connection and echoes
        // with a prefix; mirrors the CPython socket walkthrough.
        let srv = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        srv.bind(("127.0.0.1", 0)).unwrap();
        srv.listen(1).unwrap();
        let port = srv.getsockname().unwrap().1;
        let srv2 = srv.clone();
        let t = threading::Thread::new("serve", false, move || {
            let (mut conn, _addr) = srv2.accept().unwrap();
            let data = conn.recv(1024).unwrap();
            let mut reply = b"echo:".to_vec();
            reply.extend_from_slice(&data);
            conn.sendall(reply).unwrap();
            conn.close().unwrap();
            let mut srv2 = srv2;
            srv2.close().unwrap();
        });
        t.start();
        let mut cli = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        cli.connect(("127.0.0.1", port)).unwrap();
        cli.sendall(b"ping".to_vec()).unwrap();
        let got = cli.recv(1024).unwrap();
        assert_eq!(got, b"echo:ping");
        cli.close().unwrap();
        t.join();
    }

    #[test]
    fn refused_connection_raises_connection_refused_error() {
        // Verified against python3: ConnectionRefusedError('[Errno 111]
        // Connection refused'), caught by `except OSError:` through the
        // hierarchy.
        let cli = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        let e = cli.connect(("127.0.0.1", 1)).unwrap_err();
        assert_eq!(e.exception_type, "ConnectionRefusedError");
        assert!(e.matches("ConnectionError"));
        assert!(e.matches("OSError"));
        assert!(
            e.message.starts_with("[Errno "),
            "CPython message shape: {}",
            e.message
        );
    }

    #[test]
    fn recv_timeout_raises_timeout_error() {
        // Verified against python3: TimeoutError('timed out').
        let srv = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        srv.bind(("127.0.0.1", 0)).unwrap();
        srv.listen(1).unwrap();
        let port = srv.getsockname().unwrap().1;
        let cli = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        cli.connect(("127.0.0.1", port)).unwrap();
        cli.settimeout(0.05).unwrap();
        let e = cli.recv(10).unwrap_err();
        assert_eq!(format!("{}", e), "TimeoutError: timed out");
        assert!(e.matches("OSError"));
    }

    #[test]
    fn closed_socket_raises_bad_file_descriptor() {
        // Verified against python3: OSError('[Errno 9] Bad file descriptor');
        // close() through ONE handle closes every clone (object semantics).
        let mut s = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        let s2 = s.clone();
        s.close().unwrap();
        let e = s2.recv(1).unwrap_err();
        assert_eq!(format!("{}", e), "OSError: [Errno 9] Bad file descriptor");
    }

    #[test]
    fn udp_roundtrip_via_sendto_recvfrom() {
        let a = socket::socket(socket::AF_INET, socket::SOCK_DGRAM).unwrap();
        a.bind(("127.0.0.1", 0)).unwrap();
        let port = a.getsockname().unwrap().1;
        let b = socket::socket(socket::AF_INET, socket::SOCK_DGRAM).unwrap();
        b.sendto(b"datagram".to_vec(), ("127.0.0.1", port)).unwrap();
        let (data, _peer) = a.recvfrom(64).unwrap();
        assert_eq!(data, b"datagram");
    }

    #[test]
    fn bad_family_raises_os_error() {
        // Verified against python3: OSError('[Errno 97] Address family not
        // supported by protocol').
        let e = socket::socket(999, socket::SOCK_STREAM).unwrap_err();
        assert_eq!(
            format!("{}", e),
            "OSError: [Errno 97] Address family not supported by protocol"
        );
    }

    #[test]
    fn shared_socket_is_full_duplex_across_threads() {
        // Devin review round 2 on PR #144: a blocked recv() must NOT hold
        // the socket's internal lock — CPython sockets are full-duplex, so
        // a reader thread and a writer thread use ONE shared socket
        // concurrently. Pre-fix this deadlocked: the reader held the state
        // mutex across the blocking read, and the writer's sendall on the
        // same socket waited on it forever.
        let srv = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        srv.bind(("127.0.0.1", 0)).unwrap();
        srv.listen(1).unwrap();
        let port = srv.getsockname().unwrap().1;
        let cli = socket::socket(socket::AF_INET, socket::SOCK_STREAM).unwrap();
        cli.connect(("127.0.0.1", port)).unwrap();
        let (conn, _addr) = srv.accept().unwrap();
        let reader_cli = cli.clone();
        let reader = threading::Thread::new("reader", false, move || {
            // Blocks until the peer's reply — while the main thread sends
            // through ANOTHER clone of the same socket.
            let got = reader_cli.recv(16).unwrap();
            assert_eq!(got, b"pong");
        });
        reader.start();
        // Give the reader time to block inside recv() before writing.
        std::thread::sleep(std::time::Duration::from_millis(50));
        cli.sendall(b"ping".to_vec()).unwrap();
        let got = conn.recv(16).unwrap();
        assert_eq!(got, b"ping");
        conn.sendall(b"pong".to_vec()).unwrap();
        reader.join();
    }

    #[test]
    fn gethostname_is_nonempty() {
        assert!(!socket::gethostname().is_empty());
    }
}

mod bytesio {
    use stdpython::io;

    #[test]
    fn bytesio_cursor_semantics_match_python() {
        // python3: BytesIO(b"seeded").write(b"!") OVERWRITES at the
        // cursor: buffer b'!eeded', write returns 1, read() -> b'eeded'.
        let b = io::BytesIO_seeded(b"seeded");
        assert_eq!(b.write(b"!").unwrap(), 1);
        assert_eq!(b.getvalue().unwrap(), b"!eeded");
        assert_eq!(b.read().unwrap(), b"eeded");
        // At the end, write appends.
        assert_eq!(b.write(b"xy").unwrap(), 2);
        assert_eq!(b.getvalue().unwrap(), b"!eededxy");
    }

    #[test]
    fn closed_bytesio_raises_pythons_value_error() {
        // Verified against python3: ValueError('I/O operation on closed file.')
        let b = io::BytesIO();
        b.close().unwrap();
        let e = b.read().unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: I/O operation on closed file.");
    }
}

// ---- Boxed-value arithmetic (issues #115/#120) ----

#[test]
fn pyvalue_add_dispatches_like_cpython() {
    use stdpython::{PyAdd, PyValue};
    let v = |a: PyValue, b: PyValue| a.py_add(&b);
    // Verified against python3: 1 + 2 == 3; 1 + 2.5 == 3.5;
    // True + True == 2 (bool ⊂ int); 'ab' + 'cd' == 'abcd';
    // b'ab' + b'c' == b'abc'; (1, 'x') + (2,) == (1, 'x', 2).
    assert_eq!(v(PyValue::Int(1), PyValue::Int(2)), PyValue::Int(3));
    assert_eq!(v(PyValue::Int(1), PyValue::Float(2.5)), PyValue::Float(3.5));
    assert_eq!(v(PyValue::Bool(true), PyValue::Bool(true)), PyValue::Int(2));
    assert_eq!(
        v(PyValue::from("ab"), PyValue::from("cd")),
        PyValue::from("abcd")
    );
    assert_eq!(
        v(PyValue::Bytes(b"ab".to_vec()), PyValue::Bytes(b"c".to_vec())),
        PyValue::Bytes(b"abc".to_vec())
    );
    let t = |vals: Vec<PyValue>| PyValue::Tuple(std::sync::Arc::new(vals));
    assert_eq!(
        v(
            t(vec![PyValue::Int(1), PyValue::from("x")]),
            t(vec![PyValue::Int(2)])
        ),
        t(vec![PyValue::Int(1), PyValue::from("x"), PyValue::Int(2)])
    );
    // Concrete right operands box and delegate.
    assert_eq!(PyValue::from("v").py_add(&"-x"), PyValue::from("v-x"));
    assert_eq!(PyValue::Int(1).py_add(&2i64), PyValue::Int(3));
}

#[test]
#[should_panic(expected = "unsupported operand type(s) for +: 'int' and 'str'")]
fn pyvalue_add_mismatch_panics_cpythons_type_error() {
    use stdpython::{PyAdd, PyValue};
    // Verified against python3: 1 + 'x' raises
    // TypeError: unsupported operand type(s) for +: 'int' and 'str'.
    let _ = PyValue::Int(1).py_add(&PyValue::from("x"));
}

#[test]
fn range_replace_mechanics_match_python() {
    // Issue #153: Python semantics verified against python3 3.14 -
    //   xs=[0,1,2,3]; xs[1:3]=["a","b"] -> [0,"a","b",3]   (replace)
    //   xs[1:1]=["q"]                   -> insert          (zero-width)
    //   del xs[1:3]                     -> removes range
    //   strided: ys[::2]=[9,9] on [0,1,2] -> [9,1,9]
    //   mismatched stride length raises ValueError
    use stdpython::{py_value_str, PySliceReplace, PyValue};
    let mut v = vec![
        PyValue::Int(0),
        PyValue::Int(1),
        PyValue::Int(2),
        PyValue::Int(3),
    ];
    v.py_slice_assign(
        Some(1),
        Some(3),
        vec![PyValue::Str("a".into()), PyValue::Str("b".into())],
    );
    assert_eq!(
        py_value_str(&PyValue::Tuple(std::sync::Arc::new(v.clone()))),
        "(0, 'a', 'b', 3)"
    );
    v.py_slice_assign(Some(1), Some(1), vec![PyValue::Int(9)]);
    assert_eq!(v.len(), 5);
    v.py_slice_delete(Some(1), Some(3));
    assert_eq!(v.len(), 3);
    let mut ys = vec![PyValue::Int(0), PyValue::Int(1), PyValue::Int(2)];
    ys.py_slice_assign_step(None, None, 2, vec![PyValue::Int(9), PyValue::Int(9)])
        .unwrap();
    assert_eq!(ys.len(), 3);
    assert!(matches!(ys[0], PyValue::Int(9)));
    // Negative-step DELETE: extended_slice_indices walks descending, so
    // removing in emitted order is correct. Verified against python3:
    //   [1,2,3,4]; del x[::-1] -> []
    //   [0,1,2];   del x[::-2] -> [1]
    let mut rev = vec![
        PyValue::Int(1),
        PyValue::Int(2),
        PyValue::Int(3),
        PyValue::Int(4),
    ];
    rev.py_slice_delete_step(None, None, -1).unwrap();
    assert_eq!(rev.len(), 0);
    let mut rev2 = vec![PyValue::Int(0), PyValue::Int(1), PyValue::Int(2)];
    rev2.py_slice_delete_step(None, None, -2).unwrap();
    let kept: Vec<i64> = rev2
        .iter()
        .map(|v| match v {
            PyValue::Int(i) => *i,
            other => unreachable!("{other:?}"),
        })
        .collect();
    assert_eq!(kept, vec![1]);
}

// Gated on plain std (the default feature set): urllib.parse is pure
// string handling since round 57's un-gating (the retrospective's R6
// correction — these tests previously required the http-ureq feature,
// which CI does not enable, so the fidelity bugs shipped unchecked).
#[cfg(feature = "std")]
mod urllib_parse_pins {
    use stdpython::*;

#[test]
fn urllib_parse_matches_cpython() {
    use stdpython::urllib::parse::*;
    // Verified against python3 (urllib.parse, CPython 3.11):
    let p = urlparse("https://user:pass@example.com:8080/path/to?a=1&b=2#frag").unwrap();
    assert_eq!(p.scheme, "https");
    assert_eq!(p.netloc, "user:pass@example.com:8080");
    assert_eq!(p.path, "/path/to");
    assert_eq!(p.params, "");
    assert_eq!(p.query, "a=1&b=2");
    assert_eq!(p.fragment, "frag");
    assert_eq!(p.hostname(), Some("example.com".to_string()));
    assert_eq!(p.port().unwrap(), Some(8080));
    assert_eq!(p.username(), Some("user".to_string()));
    assert_eq!(p.password(), Some("pass".to_string()));
    assert_eq!(p.geturl(), "https://user:pass@example.com:8080/path/to?a=1&b=2#frag");
    // urlsplit: params empty.
    let s = urlsplit("https://example.com/a?x=1").unwrap();
    assert_eq!((s.scheme.as_str(), s.netloc.as_str(), s.path.as_str(), s.query.as_str(), s.fragment.as_str()),
               ("https", "example.com", "/a", "x=1", ""));
    // urlunparse: six components.
    assert_eq!(
        urlunparse(("https", "example.com", "/p", "", "q=1", "f")).unwrap(),
        "https://example.com/p?q=1#f"
    );
    // urljoin: relative resolution and absolute-target precedence.
    assert_eq!(urljoin("http://example.com/a/b/c", "../../d").unwrap(), "http://example.com/d");
    assert_eq!(urljoin("http://example.com/a", "https://other.com/x").unwrap(), "https://other.com/x");
    // quote: unreserved pass, everything else %XX.
    assert_eq!(quote("a b&c=d", None).unwrap(), "a%20b%26c%3Dd");
    assert_eq!(quote("a b&c=d", Some("&")).unwrap(), "a%20b&c%3Dd");
    assert_eq!(quote_plus("a b&c").unwrap(), "a+b%26c");
    // unquote: %XX decodes.
    assert_eq!(unquote("a%20b%26c").unwrap(), "a b&c");
    assert_eq!(unquote_plus("a+b%20c").unwrap(), "a b c");
    // urldefrag: split at the first #.
    assert_eq!(urldefrag("http://x.com/a#frag").unwrap(), ("http://x.com/a".to_string(), "frag".to_string()));
    assert_eq!(urldefrag("http://x.com/a").unwrap(), ("http://x.com/a".to_string(), String::new()));
    // The retrospective's R6 findings on #260, each verified against
    // python3 (CPython 3.11):
    //   urlsplit("http://example.com/p;q?x=1").path  == "/p;q" (params
    //   NOT split out; the round-55 version deleted ";q").
    let sp = urlsplit("http://example.com/p;q?x=1").unwrap();
    assert_eq!((sp.path.as_str(), sp.params.as_str()), ("/p;q", ""));
    //   urlparse("http://[::1]:8080/a").hostname() == "::1" (IPv6
    //   brackets stripped).
    let ipv6 = urlparse("http://[::1]:8080/a").unwrap();
    assert_eq!(ipv6.hostname(), Some("::1".to_string()));
    //   urlparse("http://user@name:pass@example.com/").username() ==
    //   "user@name" (the LAST @ splits userinfo from host).
    let ui = urlparse("http://user@name:pass@example.com/").unwrap();
    assert_eq!(ui.username(), Some("user@name".to_string()));
    assert_eq!(ui.password(), Some("pass".to_string()));
    // `.port` (CPython 3.11): urlparse succeeds, the read raises.
    //   urlparse("http://h:0080/").port == 80
    //   urlparse("http://h:/").port is None; urlparse("http://h/").port is None
    //   urlparse("http://[::1]:8/").port == 8
    //   urlparse("http://h:abc/").port  -> ValueError: Port could not be cast to integer value as 'abc'
    //   urlparse("http://h:-1/").port   -> ValueError: Port could not be cast to integer value as '-1'
    //   urlparse("http://h:70000/").port -> ValueError: Port out of range 0-65535
    let port = |u: &str| urlparse(u).unwrap().port().map_err(|e| e.to_string());
    assert_eq!(port("http://h:0080/"), Ok(Some(80)));
    assert_eq!(port("http://h:/"), Ok(None));
    assert_eq!(port("http://h/"), Ok(None));
    assert_eq!(port("http://[::1]:8/"), Ok(Some(8)));
    assert_eq!(
        port("http://h:abc/"),
        Err("ValueError: Port could not be cast to integer value as 'abc'".to_string())
    );
    assert_eq!(
        port("http://h:-1/"),
        Err("ValueError: Port could not be cast to integer value as '-1'".to_string())
    );
    assert_eq!(port("http://h:70000/"), Err("ValueError: Port out of range 0-65535".to_string()));
    assert_eq!(ui.hostname(), Some("example.com".to_string()));
    //   urlparse("HTTP://EXAMPLE.COM/").scheme == "http" (lowercased).
    assert_eq!(urlparse("HTTP://EXAMPLE.COM/").unwrap().scheme, "http");
}

#[test]
fn urllib_urlencode_matches_cpython() {
    use stdpython::urllib::parse::urlencode;
    use stdpython::PyValue;
    // Verified against python3:
    //   urlencode({"a": 1, "b": "x y"}) == "a=1&b=x+y"
    let mut d = PyDict::new();
    d.insert("a".to_string(), PyValue::Int(1));
    d.insert("b".to_string(), PyValue::Str("x y".to_string()));
    assert_eq!(urlencode(&PyValue::from(d), false).unwrap(), "a=1&b=x+y");
    //   urlencode([("a", 1), ("a", 2)], doseq=True) == "a=1&a=2"
    let pairs = PyValue::from(vec![
        PyValue::from(vec![PyValue::Str("a".to_string()), PyValue::Int(1)]),
        PyValue::from(vec![PyValue::Str("a".to_string()), PyValue::Int(2)]),
    ]);
    assert_eq!(urlencode(&pairs, true).unwrap(), "a=1&a=2");
}

}

// A Python TUPLE value boxes as Tuple members (round 56): `PyValue::from(
// ("str".to_string(), "bytes".to_string()))` — the module-level
// class-as-value tuples requests' compat produces (`basestring = (str,
// bytes)`, `numeric_types = (int, float)`). Verified against python3: a
// tuple of the name strings, indexed like the Python tuple.
#[test]
fn pyvalue_from_tuple_boxes_as_tuple_members() {
    use stdpython::*;
    // (str, bytes) as name strings — requests' compat basestring.
    let t: PyValue = ("str".to_string(), "bytes".to_string()).into();
    let PyValue::Tuple(members) = &t else {
        panic!("a 2-tuple must box as a Tuple, got {:?}", t);
    };
    assert_eq!(members.len(), 2);
    assert_eq!(members[0], PyValue::from("str"));
    assert_eq!(members[1], PyValue::from("bytes"));
    // A mixed tuple (String, i64) — each element converts through its
    // own Into<PyValue>.
    let mixed: PyValue = ("name".to_string(), 3).into();
    let PyValue::Tuple(members) = &mixed else {
        panic!("a mixed tuple must box as a Tuple, got {:?}", mixed);
    };
    assert_eq!(members.len(), 2);
    assert_eq!(members[0], PyValue::from("name"));
    assert_eq!(members[1], PyValue::from(3));
    // A 1-tuple (integer_types = (int,)) keeps its trailing comma shape.
    let one: PyValue = ("int".to_string(),).into();
    let PyValue::Tuple(members) = &one else {
        panic!("a 1-tuple must box as a Tuple, got {:?}", one);
    };
    assert_eq!(members.len(), 1);
    // The 6-arity impl (urlunparse-style tuple values).
    let six: PyValue = ("a", 1, 2.5, true, "b", 6i64).into();
    let PyValue::Tuple(members) = &six else {
        panic!("a 6-tuple must box as a Tuple, got {:?}", six);
    };
    assert_eq!(members.len(), 6);
}

// A BOXED list's membership test against a string (round 57): a list
// that boxes because one element is `str | None` (`encoding_iana in
// [specified_encoding, "ascii", "utf_8"]` — charset_normalizer's
// from_sequence) compares the Str members by value, exactly like Python's
// `x in [.., None-or-str, ..]` — the None element never matches.
#[test]
fn boxed_list_py_contains_matches_str_members_by_value() {
    use stdpython::*;
    let list: Vec<PyValue> = vec![
        PyValue::from("ascii"),
        PyValue::from("utf_8"),
        stdpython::PyValue::None_,
    ];
    assert!(list.py_contains(&"ascii"), "a Str member matches");
    assert!(list.py_contains(&"utf_8"), "a later Str member matches");
    assert!(!list.py_contains(&"latin1"), "an absent member does not match");
    // The String operand spelling the renderers emit for an owned name.
    let needle = "utf_8".to_string();
    assert!(list.py_contains(&needle), "an owned String operand matches");
}

// The boxed-index-by-int projection a promoted tuple-unpack uses (round
// 57, Devin review #263 Findings 3+4): `PyValue::from((1.5, 2.5))
// .py_index(i)` yields the Float members unchanged (no truncation), and
// a boxed BYTES value indexes to its Int elements. Verified against
// python3: (1.5, 2.5)[0] == 1.5; b"VMDI"[0] == 86.
#[test]
fn boxed_index_by_int_projects_unpack_elements() {
    use stdpython::*;
    let t: PyValue = (1.5, 2.5).into();
    assert_eq!(t.py_index(0i64).unwrap(), PyValue::from(1.5));
    assert_eq!(t.py_index(1i64).unwrap(), PyValue::from(2.5));
    assert!(t.py_index(2i64).is_err(), "out of range is an IndexError");
    let b: PyValue = b"VMDI".to_vec().into();
    assert_eq!(b.py_index(0i64).unwrap(), PyValue::from(86));
    assert_eq!(b.py_index(3i64).unwrap(), PyValue::from(73));
    // Negative indexes normalize to the last element (CPython
    // `b"VMDI"[-1]` == 73), and out-of-range reads raise the exact
    // per-type messages (Devin review on #263):
    //   b"ab"[5] -> IndexError('index out of range')
    //   (1, 2)[5] -> IndexError('tuple index out of range')
    //   "ab"[5] -> IndexError('string index out of range')
    assert_eq!(b.py_index(-1i64).unwrap(), PyValue::from(73));
    let err = b.py_index(4i64).unwrap_err();
    assert_eq!((err.exception_type.as_str(), err.message.as_str()), ("IndexError", "index out of range"));
    let t: PyValue = (1, 2).into();
    assert_eq!(t.py_index(-1i64).unwrap(), PyValue::from(2));
    let err = t.py_index(5i64).unwrap_err();
    assert_eq!((err.exception_type.as_str(), err.message.as_str()), ("IndexError", "tuple index out of range"));
    let st: PyValue = "ab".into();
    assert_eq!(st.py_index(-1i64).unwrap(), PyValue::from("b"));
    let err = st.py_index(5i64).unwrap_err();
    assert_eq!((err.exception_type.as_str(), err.message.as_str()), ("IndexError", "string index out of range"));
    // Not-subscriptable values raise CPython's per-type TypeError text.
    let err = PyValue::from(5i64).py_index(0i64).unwrap_err();
    assert_eq!((err.exception_type.as_str(), err.message.as_str()), ("TypeError", "'int' object is not subscriptable"));
    let err = stdpython::PyValue::None_.py_index(0i64).unwrap_err();
    assert_eq!((err.exception_type.as_str(), err.message.as_str()), ("TypeError", "'NoneType' object is not subscriptable"));
}

// A literal set's membership against an owned String operand (round 60):
// `{"utf_16", "utf_32"}` builds as HashSet<&str> and `encoding_iana in
// {...}` (charset_normalizer) passes an owned String — the generic
// PyContains<T> for HashSet<T> covers the &str spellings, and the String
// impl compares by value. Verified against python3: "utf_16" in {"utf_16"}.
#[test]
fn literal_set_py_contains_owned_string() {
    use stdpython::*;
    let s: std::collections::HashSet<&str> =
        std::collections::HashSet::from(["utf_16", "utf_32"]);
    assert!(s.py_contains(&"utf_16".to_string()), "a member String matches");
    assert!(!s.py_contains(&"latin1".to_string()), "an absent String does not");
    assert!(s.py_contains(&"utf_32"), "the &str spelling still matches");
}

// A literal list builds as Vec<&str>; membership against an owned String
// operand (urllib3's CONTENT_DECODERS constants, round 61b) resolves
// through the str/String spellings, comparing by value.
#[test]
fn literal_list_py_contains_owned_string() {
    use stdpython::*;
    let l: Vec<&str> = vec!["gzip", "deflate"];
    assert!(l.py_contains(&"gzip".to_string()), "a member String matches");
    assert!(!l.py_contains(&"br".to_string()), "an absent String does not");
    assert!(l.py_contains(&"deflate"), "the &str spelling still matches");
}

// A literal list's membership against a BOXED Str operand (`direction in
// ("R", "AL")` where direction is boxed — idna's _is_bidi, round 61b).
#[test]
fn literal_list_py_contains_boxed_str() {
    use stdpython::*;
    let l: Vec<&str> = vec!["R", "AL", "AN"];
    assert!(l.py_contains(&PyValue::from("R")), "a Str member matches");
    assert!(!l.py_contains(&PyValue::from("L")), "an absent Str does not");
    assert!(!l.py_contains(&PyValue::from(5)), "a non-Str boxed value never matches");
}

#[test]
fn py_boxed_str_ops_dispatch_on_the_runtime_member() {
    // python3: d = {"scheme": "HTTPS"}; d["scheme"].lower() == "https" —
    // the boxed member's str method dispatches on the runtime type.
    let boxed = PyValue::Str("HTTPS".to_string());
    assert_eq!(boxed.py_boxed_lower(), "https");
    assert_eq!(boxed.py_boxed_upper(), "HTTPS");
    assert_eq!(PyValue::Str("  x  ".to_string()).py_boxed_strip(), "x");
    // python3: d = {"k": 1}; d["k"].lower() raises AttributeError:
    // 'int' object has no attribute 'lower' — the loud §12.2 panic.
    let caught = std::panic::catch_unwind(|| {
        let _ = PyValue::Int(5).py_boxed_lower();
    });
    assert!(caught.is_err(), "non-str members must panic loudly");
}

#[test]
fn compiled_regex_matches_with_python_anchoring() {
    // python3: re.compile(r"a+").match("ba") is None (anchored at the
    // start); .search("ba") matches "a"; .fullmatch("aa") matches,
    // .fullmatch("ba") is None (whole text required).
    use stdpython::stdlib::re::{PyRegexOps, Regex};
    let re = Regex::new("a+").unwrap();
    assert!(re.py_match("ba").is_none(), "match anchors at the start");
    assert_eq!(re.py_match("aaa").unwrap().groups(), Vec::<String>::new());
    assert_eq!(re.py_search("ba").unwrap().groups(), Vec::<String>::new());
    assert!(re.py_fullmatch("ba").is_none(), "fullmatch requires the whole text");
    assert!(re.py_fullmatch("aaa").is_some());
    // A capturing pattern's groups() carries the groups.
    let re2 = Regex::new("^([^?#]*)(?:\\?([^#]*))?.*$").unwrap();
    let m = re2.py_match("path?query").unwrap();
    assert_eq!(m.groups(), vec!["path", "query"]);
    // python3: re.fullmatch("a|ab", "ab") is a match ("ab") — the
    // engine is CONSTRAINED to the whole string, so the alternation
    // resolves to "ab" even though the unanchored leftmost-first match
    // is "a". A post-hoc filter of the unanchored match would return
    // None here.
    let alt = Regex::new("a|ab").unwrap();
    assert_eq!(
        alt.py_fullmatch("ab").unwrap().groups(),
        Vec::<String>::new(),
        "fullmatch must let the alternation resolve to the whole-string branch"
    );
    assert!(alt.py_fullmatch("a").is_some(), "whole text 'a' matches");
    assert!(alt.py_fullmatch("b").is_none(), "'b' does not match");
    // python3: re.fullmatch("a*?", "aaa") matches "aaa" — the lazy
    // quantifier must expand until the whole text is covered.
    let lazy = Regex::new("a*?").unwrap();
    assert!(
        lazy.py_fullmatch("aaa").is_some(),
        "a lazy quantifier must expand to cover the whole text"
    );
    // python3: re.fullmatch("a|ab", "AB", re.IGNORECASE) matches "AB".
    let ci = stdpython::stdlib::re::compile("a|ab", "i").unwrap();
    assert!(ci.py_fullmatch("AB").is_some(), "flags carry into fullmatch");
    // Groups are preserved through the anchored engine: python3
    // re.fullmatch("(a)|(ab)", "ab") gives group(0)="ab", group(1)=None
    // (a non-participating group — a loud ValueError in rython's typed
    // lowering), group(2)="ab".
    let grp = Regex::new("(a)|(ab)").unwrap();
    let m = grp.py_fullmatch("ab").unwrap();
    assert_eq!(m.group(0), "ab");
    assert_eq!(m.group(2), "ab");
    // python3: re.match("a(b)?", "a").span(1) is (-1, -1) — a
    // non-participating group's span is the sentinel pair, not an error.
    let opt = Regex::new("a(b)?").unwrap();
    let om = opt.py_match("a").unwrap();
    assert_eq!(om.span_group(1), (-1, -1), "span of an absent group is (-1, -1)");
    assert_eq!(om.span_group(0), (0, 1));
}

#[test]
fn string_is_family_matches_python() {
    // python3 ground truth (ASCII): "ABC" isupper=True islower=False
    // isalpha=True isdigit=False isdecimal=False isalnum=True
    // isspace=False isprintable=True istitle=False; "A1" istitle=True;
    // "" isprintable=True but everything else False; "  " isspace=True;
    // "Hello World" istitle=True; "HELLO" istitle=False.
    use stdpython::PyStrOps;
    let cases: [(&str, bool, bool, bool, bool, bool, bool, bool, bool, bool); 11] = [
        ("ABC", true, false, true, false, false, true, false, true, false),
        ("abc", false, true, true, false, false, true, false, true, false),
        ("A1", true, false, false, false, false, true, false, true, true),
        ("123", false, false, false, true, true, true, false, true, false),
        ("", false, false, false, false, false, false, false, true, false),
        ("  ", false, false, false, false, false, false, true, true, false),
        ("Hello World", false, false, false, false, false, false, false, true, true),
        ("HELLO", true, false, true, false, false, true, false, true, false),
        ("abc123", false, true, false, false, false, true, false, true, false),
        ("a b", false, true, false, false, false, false, false, true, false),
        ("3rd", false, true, false, false, false, true, false, true, false),
    ];
    for (s, up, lo, al, di, de, an, sp, pr, ti) in cases {
        assert_eq!(s.isupper(), up, "isupper({s:?})");
        assert_eq!(s.islower(), lo, "islower({s:?})");
        assert_eq!(s.isalpha(), al, "isalpha({s:?})");
        assert_eq!(s.isdigit(), di, "isdigit({s:?})");
        assert_eq!(s.isdecimal(), de, "isdecimal({s:?})");
        assert_eq!(s.isalnum(), an, "isalnum({s:?})");
        assert_eq!(s.isspace(), sp, "isspace({s:?})");
        assert_eq!(s.isprintable(), pr, "isprintable({s:?})");
        assert_eq!(s.istitle(), ti, "istitle({s:?})");
    }
    // python3: the four separator controls are whitespace.
    for cp in 0x1C..=0x1Fu32 {
        let c = char::from_u32(cp).unwrap();
        assert!(c.to_string().isspace(), "U+{cp:04X} isspace");
    }
    // python3: non-ASCII spaces, line separators, format characters, and
    // unassigned code points are NOT printable; ASCII space and tab are.
    for cp in [0xA0u32, 0x200B, 0x2028, 0x2029, 0xAD, 0xFEFF, 0x2060, 0x378] {
        let c = char::from_u32(cp).unwrap();
        assert!(!c.to_string().isprintable(), "U+{cp:04X} isprintable");
    }
    assert!(" ".isprintable(), "ASCII space is printable");
    // python3: control characters (tab, newline, NUL) are NOT printable.
    assert!(!'\t'.to_string().isprintable());
    assert!(!'\n'.to_string().isprintable());
    assert!(!'\u{0}'.to_string().isprintable());
    // python3: str.isascii() — true for the empty string and when every
    // character is 7-bit ASCII, false for any non-ASCII character.
    assert!("".isascii(), "empty string isascii");
    assert!("abc123 \t\n".isascii(), "ASCII text isascii");
    assert!(!'\u{00E9}'.to_string().isascii(), "é is not ASCII");
    assert!(!'\u{00FF}'.to_string().isascii(), "ÿ is not ASCII");
    assert!(!'\u{20AC}'.to_string().isascii(), "€ is not ASCII");
    // python3: U+0345 (a combining mark with the Alphabetic property) is
    // not a letter — isalpha/isalnum are False; A-grave is a letter.
    assert!(!'\u{0345}'.to_string().isalpha());
    assert!(!'\u{0345}'.to_string().isalnum());
    assert!('\u{00C0}'.to_string().isalpha(), "\u{00C0} isalpha");
}

#[test]
fn boxed_values_convert_back_to_typed_members() {
    // Round 80: a boxed PyValue flows back into a typed slot or
    // `impl Into<T>` parameter via the reverse From impls — the value
    // was boxed from a concrete member, so the conversion recovers it;
    // a wrong member is a LOUD TypeError panic (Python fails at use,
    // rython at the conversion).
    use stdpython::PyValue;
    let s: String = PyValue::from("abc").into();
    assert_eq!(s, "abc");
    let b: Vec<u8> = PyValue::from(b"xy".to_vec()).into();
    assert_eq!(b, b"xy");
    let i: i64 = PyValue::from(7).into();
    assert_eq!(i, 7);
    let f: f64 = PyValue::from(1.5).into();
    assert_eq!(f, 1.5);
    let bl: bool = PyValue::from(true).into();
    assert!(bl);
    // A wrong member is loud.
    let v = PyValue::from(3);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: String = v.into();
    }));
    assert!(r.is_err(), "a non-str boxed value into a String slot must panic");
}

#[test]
fn list_index_and_count_match_python() {
    // list.index(x) returns the first equal index; list.count(x) the
    // occurrences — verified against python3.
    let v = vec!["a".to_string(), "b".to_string(), "a".to_string()];
    assert_eq!(v.py_index_of(&"b".to_string()).unwrap(), 1);
    assert_eq!(v.py_index_of(&"a".to_string()).unwrap(), 0);
    // CPython 3.14's exact messages (verified against python3.14.1 —
    // the modern 3.10+ form, NOT the pre-3.10 "X is not in list").
    let err = v.py_index_of(&"z".to_string()).unwrap_err();
    assert_eq!(err.message, "list.index(x): x not in list");
    let mut d = stdpython::collections::deque::<String>::new();
    d.append("a".to_string());
    let derr = d.py_index_of(&"z".to_string()).unwrap_err();
    assert_eq!(derr.message, "deque.index(x): x not in deque");
    assert_eq!(v.count(&"a".to_string()), 2);
    assert_eq!(v.count(&"z".to_string()), 0);
}

#[test]
fn exception_attrs_and_key_repr_match_python() {
    // The exception-attribute model: a constructed exception carries its
    // __init__ fields (attr_i64) and its ancestor chain (matches on the
    // base class); KeyError's key repr is single-quoted — verified
    // against python3.
    let e = stdpython::PyException::new_with_attrs_and_ancestors(
        "InsufficientFunds",
        "need 100, have 30",
        vec![
            ("needed".to_string(), stdpython::PyValue::from(100i64)),
            ("available".to_string(), stdpython::PyValue::from(30i64)),
        ],
        vec!["BankError".to_string(), "Exception".to_string()],
    );
    assert_eq!(e.attr_i64("needed").unwrap(), 100);
    assert_eq!(e.attr_i64("available").unwrap(), 30);
    assert!(e.attr_i64("other").is_err(), "an absent field raises AttributeError");
    assert!(e.matches("BankError"), "the ancestor chain reaches the base class");
    assert_eq!(
        stdpython::key_repr(&"carol".to_string()),
        "'carol'"
    );
    assert_eq!(stdpython::key_repr(&7i64), "7");
    let k: String = "carol".into();
    let err = stdpython::PyDict::<String, i64>::new()
        .py_index(k)
        .unwrap_err();
    assert_eq!(err.message, "'carol'", "KeyError quotes a str key like CPython");
}

/// A raised USER class derived from a builtin (`class MyError(ValueError)`)
/// is caught by `except ValueError:` — the discriminant fast path consults
/// the ancestor chain the construction attached (the evaluation on issue
/// #137). Verified against python3:
///   class MyError(ValueError): pass
///   try: raise MyError("x")
///   except ValueError: print("caught")   -> caught
#[test]
fn a_user_exception_over_a_builtin_is_caught_by_the_builtin_handler() {
    use stdpython::{BuiltinException as B, PyException};
    let e = PyException::new_with_attrs_and_ancestors(
        "MyError",
        "x",
        vec![],
        vec!["ValueError".to_string(), "Exception".to_string()],
    );
    assert!(e.matches_builtin(B::ValueError));
    assert!(e.matches_builtin(B::Exception));
    assert!(e.matches_builtin(B::BaseException));
    assert!(!e.matches_builtin(B::KeyError), "siblings do not catch");
    assert!(!e.matches_builtin(B::LookupError));
}

#[test]
fn a_user_exception_over_base_exception_is_not_an_exception() {
    // `class Exit(BaseException)`: the recorded chain decides — caught by
    // `except BaseException:`, not by `except Exception:` (Devin review
    // on #330). A chain with no builtin in it keeps the broad posture.
    use stdpython::{BuiltinException as B, PyException};
    let e = PyException::new_with_attrs_and_ancestors(
        "Exit",
        "bye",
        vec![],
        vec!["BaseException".to_string()],
    );
    assert!(e.matches_builtin(B::BaseException));
    assert!(!e.matches_builtin(B::Exception));
    assert!(e.matches("BaseException"));
    assert!(!e.matches("Exception"));
    let unknown = PyException::new_with_attrs_and_ancestors(
        "Odd",
        "x",
        vec![],
        vec!["SomeImportedError".to_string()],
    );
    assert!(unknown.matches_builtin(B::Exception));
    assert!(unknown.matches("Exception"));
    assert!(!unknown.matches_builtin(B::ValueError));
}

#[test]
fn an_exceptions_repr_is_its_class_and_its_args() {
    // CPython's BaseException.__repr__: the class name and the args
    // tuple's contents, from the recorded reprs; a construction that
    // recorded none quotes its message, or is `Kind()` when empty (Devin
    // review on #330).
    use stdpython::{PyException, PyRepr};
    let e = PyException::new_with_attrs_and_ancestors("Inner", "('a', 'b')", vec![], vec![])
        .with_args_repr(vec!["'a'".to_string(), "'b'".to_string()]);
    assert_eq!(e.py_repr(), "Inner('a', 'b')");
    let one = PyException::new("Inner", "x");
    assert_eq!(one.py_repr(), "Inner('x')");
    let none = PyException::new("Inner", "");
    assert_eq!(none.py_repr(), "Inner()");
}

#[test]
fn file_errors_quote_paths_as_python_does_and_wrong_direction_io_is_unsupported_operation() {
    // Round 4 of the review on #339: an OSError's filename is Python's
    // repr (an apostrophe switches to double quotes); a write on a
    // read-only stream or a read on a write-only one is
    // io.UnsupportedOperation — an OSError AND a ValueError in CPython's
    // tree — with CPython's messages (`write`/`read` for a binary file,
    // `not writable`/`not readable` for a text file); the standard
    // streams are one shared object each, so closing one alias closes
    // the other.
    let err = stdpython::open("/nonexistent/dir/it's.txt", Some("r")).err().expect("missing");
    assert!(err.matches("FileNotFoundError"), "{:?}", err);
    assert_eq!(
        err.message,
        "[Errno 2] No such file or directory: \"/nonexistent/dir/it's.txt\""
    );
    let dir = std::env::temp_dir().join(format!("rython-uo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("x.bin");
    std::fs::write(&path, b"abc").unwrap();
    let p = path.to_str().unwrap();
    let err = stdpython::open_binary(p, "rb").unwrap().write(b"x").err().expect("read-only");
    assert!(err.matches("UnsupportedOperation") && err.matches("OSError") && err.matches("ValueError"), "{:?}", err);
    assert_eq!(err.message, "write");
    let err = stdpython::open_binary(p, "wb").unwrap().read().err().expect("write-only");
    assert!(err.matches("OSError"), "{:?}", err);
    assert_eq!(err.message, "read");
    let err = stdpython::open(p, Some("r")).unwrap().write("x").err().expect("read-only text");
    assert!(err.matches("UnsupportedOperation"), "{:?}", err);
    assert_eq!(err.message, "not writable");
    let err = stdpython::open(p, Some("w")).unwrap().read().err().expect("write-only text");
    assert_eq!(err.message, "not readable");
    // A post-open I/O failure is the OSError CPython raises (round 8):
    // /dev/full accepts the buffered write and fails the flush with
    // `[Errno 28] No space left on device`, text and binary alike.
    if std::path::Path::new("/dev/full").exists() {
        let full = stdpython::open_binary("/dev/full", "wb").unwrap();
        full.write(b"x").unwrap();
        let err = full.flush().err().expect("no space");
        assert!(err.matches("OSError"), "{:?}", err);
        assert_eq!(err.message, "[Errno 28] No space left on device");
        let full = stdpython::open("/dev/full", Some("w")).unwrap();
        full.write("x").unwrap();
        let err = full.flush().err().expect("no space");
        assert!(err.matches("OSError"), "{:?}", err);
        assert_eq!(err.message, "[Errno 28] No space left on device");
    }
    // getvalue() on a disk handle is CPython's AttributeError, naming the
    // receiver's class (round 5).
    let err = stdpython::open_binary(p, "rb").unwrap().getvalue().err().expect("no getvalue");
    assert!(err.matches("AttributeError"), "{:?}", err);
    assert_eq!(err.message, "'_io.BufferedReader' object has no attribute 'getvalue'");
    let err = stdpython::open_binary(p, "wb").unwrap().getvalue().err().expect("no getvalue");
    assert_eq!(err.message, "'_io.BufferedWriter' object has no attribute 'getvalue'");
    let err = stdpython::open(p, Some("r")).unwrap().getvalue().err().expect("no getvalue");
    assert_eq!(err.message, "'_io.TextIOWrapper' object has no attribute 'getvalue'");
    std::fs::remove_dir_all(&dir).unwrap();
    // ONE stdin and ONE stdout object, process-wide (round 5): the text
    // and the binary handles are aliases of the same stream, and a close
    // on another thread is seen here — CPython's `sys.stdin`/`sys.stdout`
    // (whose text wrapper and binary buffer close together).
    let text_in = stdpython::PyFile::stdin();
    assert!(!text_in.closed());
    stdpython::io::PyBytesIO::stdin().close().unwrap();
    assert!(text_in.closed(), "the binary alias closed the one stdin");
    assert_eq!(text_in.read().err().expect("closed").message, "I/O operation on closed file.");
    let a = stdpython::PyFile::stdout();
    let b = stdpython::io::PyBytesIO::stdout();
    assert!(!a.closed() && !b.closed());
    std::thread::spawn(|| stdpython::PyFile::stdout().close().unwrap())
        .join()
        .unwrap();
    assert!(a.closed() && b.closed(), "a close on another thread closes every alias");
    assert_eq!(b.write(b"x").err().expect("closed").message, "I/O operation on closed file.");
    // print() sees the same closed stdout; the embedder's reset reopens
    // both streams for the next program (round 7).
    assert_eq!(
        stdpython::print_parts(&["x"], " ", "\n").err().expect("closed stdout").message,
        "I/O operation on closed file."
    );
    stdpython::reopen_standard_streams();
    assert!(!a.closed() && !b.closed() && !text_in.closed());
    stdpython::print_parts(&[] as &[&str], "", "").unwrap();
}

#[test]
fn invalid_utf8_in_a_text_file_is_cpythons_unicode_decode_error() {
    // Round 9 of the review on #339: a text-mode read of invalid UTF-8 is
    // UnicodeDecodeError, never an OSError, with CPython's message — one
    // byte (`can't decode byte 0xff in position 2: invalid start byte`)
    // or a run (`bytes in position 0-1: unexpected end of data`). The
    // reader decodes 8192-byte chunks as TextIOWrapper does, so
    // readline() on a small file fails on its first call even when the
    // bad byte sits on a later line, with the same position; the seven
    // cases are python3 3.11's messages verbatim.
    let dir = std::env::temp_dir().join(format!("rython-utf8-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.txt");
    let p = path.to_str().unwrap();
    let cases: &[(&[u8], &str)] = &[
        (b"\xff", "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"),
        (b"ab\xffcd", "'utf-8' codec can't decode byte 0xff in position 2: invalid start byte"),
        (b"\xe2\x82", "'utf-8' codec can't decode bytes in position 0-1: unexpected end of data"),
        (b"\xe2\x82x", "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte"),
        (b"ab\xc0\xafz", "'utf-8' codec can't decode byte 0xc0 in position 2: invalid start byte"),
        (b"\xed\xa0\x80", "'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte"),
        (b"ok\n\xffbad\n", "'utf-8' codec can't decode byte 0xff in position 3: invalid start byte"),
    ];
    for (bytes, expected) in cases {
        std::fs::write(&path, bytes).unwrap();
        let err = stdpython::open(p, Some("r")).unwrap().read().err().expect("read fails");
        assert!(err.matches("UnicodeDecodeError") && !err.matches("OSError"), "{:?}", err);
        assert_eq!(err.message, *expected, "read of {:?}", bytes);
        let err = stdpython::open(p, Some("r")).unwrap().readline().err().expect("readline fails");
        assert!(err.matches("UnicodeDecodeError"), "{:?}", err);
        assert_eq!(err.message, *expected, "readline of {:?}", bytes);
    }
    // Valid text still reads line by line, a multi-byte character across
    // the chunk boundary included.
    let mut big = "a".repeat(8191).into_bytes();
    big.extend_from_slice("é\nsecond\n".as_bytes());
    std::fs::write(&path, &big).unwrap();
    let f = stdpython::open(p, Some("r")).unwrap();
    let first = f.readline().unwrap();
    assert_eq!(first.chars().count(), 8193);
    assert!(first.ends_with("é\n"));
    assert_eq!(f.readline().unwrap(), "second\n");
    assert_eq!(f.readline().unwrap(), "");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn open_binary_validates_the_mode_before_touching_the_path() {
    // CPython's mode grammar, in its order, before the path is looked
    // at (Devin review on #339, rounds 1 and 2): a letter outside
    // `axrwb+t` or a repeated letter is `invalid mode`; `t` with `b` is
    // "can't have text and binary mode at once" (`rbt` is NOT a valid
    // Python mode); more than one access letter, or none, are the two
    // "exactly one of create/read/write/append" errors; an update mode
    // is rython's own loud refusal. "rb" and "br" are the same read.
    let missing = "/nonexistent/dir/for/rython/x.bin";
    for (mode, expected) in [
        ("rbb", "invalid mode: 'rbb'"),
        ("bbr", "invalid mode: 'bbr'"),
        ("rbq", "invalid mode: 'rbq'"),
        ("rbt", "can't have text and binary mode at once"),
        ("rwb", "must have exactly one of create/read/write/append mode"),
        ("b", "Must have exactly one of create/read/write/append mode and at most one plus"),
        ("rb+", "file mode 'rb+' is not supported yet (update modes)"),
    ] {
        let err = stdpython::open_binary(missing, mode).err().expect("refused");
        assert!(err.matches("ValueError"), "{}: {:?}", mode, err);
        assert_eq!(err.message, expected, "{}", mode);
    }
    for (mode, expected) in [
        ("rt", "[Errno 2] No such file or directory: '/nonexistent/dir/for/rython/x.bin'"),
        ("rr", "invalid mode: 'rr'"),
        ("rw", "must have exactly one of create/read/write/append mode"),
        ("t", "Must have exactly one of create/read/write/append mode and at most one plus"),
        ("r+", "file mode 'r+' is not supported yet (update modes)"),
    ] {
        let err = stdpython::open(missing, Some(mode)).err().expect("refused");
        assert_eq!(err.message, expected, "{}", mode);
    }
    let dir = std::env::temp_dir().join(format!("rython-open-binary-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("x.bin");
    std::fs::write(&path, b"abc").unwrap();
    for mode in ["rb", "br"] {
        let f = stdpython::open_binary(path.to_str().unwrap(), mode).expect(mode);
        assert_eq!(f.read().unwrap(), b"abc");
    }
    // `x` creates exclusively: FileExistsError the second time.
    let fresh = dir.join("fresh.bin");
    let f = stdpython::open_binary(fresh.to_str().unwrap(), "xb").expect("xb creates");
    f.write(b"new").unwrap();
    f.close().unwrap();
    assert_eq!(std::fs::read(&fresh).unwrap(), b"new");
    let err = stdpython::open_binary(fresh.to_str().unwrap(), "xb").err().expect("exists");
    assert!(err.matches("FileExistsError"), "{:?}", err);
    assert!(err.message.starts_with("[Errno 17] File exists: "), "{:?}", err);
    std::fs::remove_dir_all(&dir).unwrap();
}

// ============================================================================
// complex (issue #366) — pinned against CPython 3.14.1
// ============================================================================

#[test]
fn complex_repr_matches_cpython() {
    use stdpython::PyToString;
    use stdpython::PyRepr;
    // Verified against python3 3.14.1 (repr is str for complex).
    let c = |re, im| Complex::new(re, im);
    assert_eq!(c(1.0, 2.0).py_repr(), "(1+2j)");
    assert_eq!(c(0.0, 1.0).py_repr(), "1j");
    assert_eq!(c(0.0, 2.0).py_repr(), "2j");
    assert_eq!(c(-0.0, -1.0).py_repr(), "(-0-1j)");
    assert_eq!(c(0.0, -1.0).py_repr(), "-1j");
    assert_eq!(c(-1.0, 2.0).py_repr(), "(-1+2j)");
    assert_eq!(c(1.0, -2.0).py_repr(), "(1-2j)");
    assert_eq!(c(0.5, 0.25).py_repr(), "(0.5+0.25j)");
    assert_eq!(c(0.0, 0.0).py_repr(), "0j");
    assert_eq!(c(0.0, -0.0).py_repr(), "-0j");
    assert_eq!(c(3.0, 0.0).py_repr(), "(3+0j)");
    assert_eq!(c(1.0, 1.0).py_repr(), "(1+1j)");
    assert_eq!(c(1e16, 1.0).py_repr(), "(1e+16+1j)");
    assert_eq!(c(0.1, 0.2).py_repr(), "(0.1+0.2j)");
    assert_eq!(c(100.0, 0.0).py_repr(), "(100+0j)");
    assert_eq!(c(1234567.0, 1234567.0).py_repr(), "(1234567+1234567j)");
    // str is repr for complex.
    assert_eq!(c(1.0, 2.0).py_str(), "(1+2j)");
    assert_eq!(c(0.0, 2.5).py_str(), "2.5j");
}

#[test]
fn complex_bool_and_eq_matches_cpython() {
    use stdpython::PyBool;
    assert!(!Complex::new(0.0, 0.0).py_bool());
    assert!(!Complex::new(0.0, -0.0).py_bool());
    assert!(Complex::new(1.0, 2.0).py_bool());
    assert!(Complex::new(0.0, 1.0).py_bool());
    // PartialEq: complex-vs-complex equality by re & im.
    assert_eq!(Complex::new(1.0, 2.0), Complex::new(1.0, 2.0));
    assert_ne!(Complex::new(1.0, 2.0), Complex::new(2.0, 1.0));
    assert_ne!(Complex::new(0.0, 1.0), Complex::new(0.0, 2.0));
}

#[test]
fn complex_arithmetic_matches_cpython() {
    use stdpython::{PyAdd, PySub, PyMul};
    // (1+2j) + (3+4j) = (4+6j)
    assert_eq!(
        Complex::new(1.0, 2.0).py_add(&Complex::new(3.0, 4.0)),
        Complex::new(4.0, 6.0)
    );
    // (1+2j) - (3-1j) = (-2+3j)
    assert_eq!(
        Complex::new(1.0, 2.0).py_sub(&Complex::new(3.0, -1.0)),
        Complex::new(-2.0, 3.0)
    );
    // (1+2j) * (3+4j) = (-5+10j)
    assert_eq!(
        Complex::new(1.0, 2.0).py_mul(&Complex::new(3.0, 4.0)),
        Complex::new(-5.0, 10.0)
    );
    // conjugate
    assert_eq!(Complex::new(3.0, 4.0).conjugate(), Complex::new(3.0, -4.0));
}

#[test]
fn complex_accessors() {
    let z = Complex::new(3.0, 4.0);
    assert_eq!(z.re(), 3.0);
    assert_eq!(z.im(), 4.0);
    // Public fields: codegen lowers `z.real` / `z.imag` to field access.
    assert_eq!(z.real, 3.0);
    assert_eq!(z.imag, 4.0);
}

#[test]
fn complex_division_matches_cpython() {
    use stdpython::PyDiv;
    // (1+2j)/(3+4j) = 0.44+0.08j
    let q = Complex::new(1.0, 2.0).py_div(&Complex::new(3.0, 4.0)).unwrap();
    assert_eq!(q.re(), 0.44);
    assert_eq!(q.im(), 0.08);
    // (1+2j)/2 == (0.5+1j)
    let q2 = Complex::new(1.0, 2.0).py_div(&Complex::new(2.0, 0.0)).unwrap();
    assert_eq!(q2.re(), 0.5);
    assert_eq!(q2.im(), 1.0);
    // 1j/0 == ZeroDivisionError: division by zero
    let err = Complex::new(0.0, 1.0).py_div(&Complex::new(0.0, 0.0)).err().unwrap();
    assert!(err.matches("ZeroDivisionError"), "{:?}", err);
    assert_eq!(err.message, "division by zero");
    // 1j/0j also raises (0j IS complex(0,0))
    let err2 = Complex::new(0.0, 1.0).py_div(&Complex::new(0.0, 0.0)).err().unwrap();
    assert!(err2.matches("ZeroDivisionError"), "{:?}", err2);
}

#[test]
fn complex_abs_matches_cpython() {
    use stdpython::PyAbs;
    // abs(3+4j) == 5.0, abs(0j) == 0.0 (an f64)
    assert_eq!(Complex::new(3.0, 4.0).py_abs(), 5.0);
    assert_eq!(Complex::new(0.0, 0.0).py_abs(), 0.0);
    assert_eq!(Complex::new(1.0, 1.0).py_abs(), 2.0f64.sqrt());
}

#[test]
fn complex_cross_type_arithmetic_matches_cpython() {
    use stdpython::{PyAdd, PySub, PyMul};
    // 1 - 3.5j == (1-3.5j)   (i64 receiver, Complex rhs)
    let r1 = (1i64).py_sub(&Complex::new(0.0, 3.5));
    assert_eq!(r1, Complex::new(1.0, -3.5));
    // 3j - 1 == (-1+3j)      (Complex receiver, i64 rhs)
    let r2 = Complex::new(0.0, 3.0).py_sub(&1i64);
    assert_eq!(r2, Complex::new(-1.0, 3.0));
    // 2 * (1+2j) == (2+4j)   (i64 receiver, Complex rhs)
    let r3 = (2i64).py_mul(&Complex::new(1.0, 2.0));
    assert_eq!(r3, Complex::new(2.0, 4.0));
    // (1+2j) * 2 == (2+4j)   (Complex receiver, i64 rhs)
    let r4 = Complex::new(1.0, 2.0).py_mul(&2i64);
    assert_eq!(r4, Complex::new(2.0, 4.0));
    // 1.5 + (2+3j) == (3.5+3j)  (f64 receiver, Complex rhs)
    let r5 = (1.5f64).py_add(&Complex::new(2.0, 3.0));
    assert_eq!(r5, Complex::new(3.5, 3.0));
    // (2+3j) + 1.5 == (3.5+3j)  (Complex receiver, f64 rhs)
    let r6 = Complex::new(2.0, 3.0).py_add(&1.5f64);
    assert_eq!(r6, Complex::new(3.5, 3.0));
    // 1j + 2 == (2+1j)
    let r7 = Complex::new(0.0, 1.0).py_add(&2i64);
    assert_eq!(r7, Complex::new(2.0, 1.0));
}

#[test]
fn str_format_with_runtime_kwargs_matches_cpython() {
    // issue #368: the runtime `str.format(**kwargs)` path for templates
    // that cannot be resolved at conversion time. Verified against CPython
    // 3.14.1.
    let kwargs = PyDict::<String, String>::from([
        ("year".to_string(), "Y".to_string()),
        ("month".to_string(), "M".to_string()),
        ("encoding".to_string(), "utf-8".to_string()),
    ]);
    assert_eq!(
        str_format_kwargs("hello {year} and {month}", &kwargs).unwrap(),
        "hello Y and M"
    );
    // `{{`/`}}` escape to a literal brace.
    assert_eq!(
        str_format_kwargs("{{lit}} {year}", &kwargs).unwrap(),
        "{lit} Y"
    );
    // Missing key raises KeyError with the quoted name.
    let err = str_format_kwargs("a {missing} b", &kwargs).err().unwrap();
    assert!(err.matches("KeyError"), "{:?}", err);
    assert_eq!(err.message, "'missing'");
}

#[test]
fn str_format_runtime_matches_cpython() {
    // issue #368: the runtime `str.format` for templates the conversion
    // cannot see. Arguments arrive rendered through str(). Verified
    // against python3 3.11.
    let args = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let kw = |xs: &[(&str, &str)]| {
        xs.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Vec<_>>()
    };
    // "{}={}".format("k", 7) == 'k=7'; "{1}:{0}".format("k", 7) == '7:k'
    assert_eq!(str_format_runtime("{}={}", &args(&["k", "7"]), &[]).unwrap(), "k=7");
    assert_eq!(str_format_runtime("{1}:{0}", &args(&["k", "7"]), &[]).unwrap(), "7:k");
    // "{0}{0}".format("x") == 'xx'; "{{}}{}".format(3) == '{}3'
    assert_eq!(str_format_runtime("{0}{0}", &args(&["x"]), &[]).unwrap(), "xx");
    assert_eq!(str_format_runtime("{{}}{}", &args(&["3"]), &[]).unwrap(), "{}3");
    // "{!s}".format(4) == '4'; "{who} has {n}".format(who="ann", n=3)
    assert_eq!(str_format_runtime("{!s}", &args(&["4"]), &[]).unwrap(), "4");
    assert_eq!(
        str_format_runtime("{who} has {n}", &[], &kw(&[("who", "ann"), ("n", "3")])).unwrap(),
        "ann has 3"
    );
    let err = |t: &str, a: &[&str], k: &[(&str, &str)]| {
        let e = str_format_runtime(t, &args(a), &kw(k)).err().unwrap();
        (e.exception_type.clone(), e.message.clone())
    };
    // '{} {}'.format(1) -> IndexError
    assert_eq!(
        err("{} {}", &["1"], &[]),
        ("IndexError".into(), "Replacement index 1 out of range for positional args tuple".into())
    );
    // "{a}".format() -> KeyError: 'a'
    assert_eq!(err("{a}", &[], &[]), ("KeyError".into(), "'a'".into()));
    // "{}{0}".format(1, 2) / "{0}{}".format(1, 2) -> ValueError
    assert_eq!(
        err("{}{0}", &["1", "2"], &[]).1,
        "cannot switch from automatic field numbering to manual field specification"
    );
    assert_eq!(
        err("{0}{}", &["1", "2"], &[]).1,
        "cannot switch from manual field specification to automatic field numbering"
    );
    // "}" / "a{" / "{x" -> ValueError, three different messages
    assert_eq!(err("}", &[], &[]).1, "Single '}' encountered in format string");
    assert_eq!(err("a{", &[], &[]).1, "Single '{' encountered in format string");
    assert_eq!(err("{x", &[], &[("x", "1")]).1, "expected '}' before end of string");
    // "{a}".format(**{"a": 1}, a=2) -> TypeError
    assert_eq!(
        err("{a}", &[], &[("a", "1"), ("a", "2")]),
        (
            "TypeError".into(),
            "str.format() got multiple values for keyword argument 'a'".into()
        )
    );
    // A spec, a !r conversion, or {a.b}/{a[0]} is refused, not approximated.
    for t in ["{:>3}", "{!r}", "{0.x}", "{0[0]}"] {
        assert_eq!(err(t, &["1"], &[]).0, "ValueError", "{}", t);
    }
}

#[test]
fn a_forwarded_kwargs_bag_is_empty_or_refused() {
    // issue #368: `wrap(text, width, **kwargs)` models no keyword options,
    // so the forwarded bag is checked where it is used. An empty bag is a
    // no-op (the call is exactly CPython's); an option raises rather than
    // being ignored.
    let empty = PyDict::<String, PyValue>::from([]);
    assert!(refuse_forwarded_kwargs("wrap", &empty).is_ok());
    let bag = PyDict::<String, PyValue>::from([(
        "initial_indent".to_string(),
        PyValue::from("> "),
    )]);
    let err = refuse_forwarded_kwargs("wrap", &bag).err().unwrap();
    assert_eq!(err.exception_type, "NotImplementedError");
    assert!(err.message.contains("'initial_indent'"), "{}", err.message);
}

#[test]
fn csv_reader_quote_none_with_escapechar_matches_cpython() {
    use stdpython::stdlib::csv::reader;
    // Verified against CPython 3.14.1: QUOTE_NONE + escapechar makes the
    // delimiter/quote/newline literal (escapechar dropped).
    let r = reader(&["a\\,b,c\\\"d,e\\\nf"], b',', true, Some::<u8>(b'\\')).unwrap();
    assert_eq!(r[0], vec!["a,b", "c\"d", "e\nf"]);
    // QUOTE_NONE WITHOUT escapechar: `"` is literal data.
    let r2 = reader(&["\"plain\",\"has,comma\""], b',', true, None).unwrap();
    assert_eq!(r2[0], vec!["\"plain\"", "\"has", "comma\""]);
    // Default (quoting on): quotes strip.
    let r3 = reader(&["\"a,b\",c"], b',', false, None).unwrap();
    assert_eq!(r3[0], vec!["a,b", "c"]);
}

#[test]
fn unittest_assert_almost_eq_matches_cpython() {
    // Verified against CPython 3.14.1 (`assertAlmostEqual`/`assertNotAlmostEqual`).
    // `PyValue` boxes ints/floats; `places` is an i64 (default 7), `delta` an
    // `Option<PyValue>` (runtime coerces to float).
    use stdpython::stdlib::unittest::{assert_almost_eq, assert_not_almost_eq};
    use stdpython::PyValue;

    // Equal values pass regardless of places.
    assert_almost_eq(&PyValue::from(1.0), &PyValue::from(1.0), 7, &None, "".to_string()).unwrap();
    // A tiny diff passes at the default 7 places.
    assert_almost_eq(
        &PyValue::from(1.0),
        &PyValue::from(1.0 + 1e-12),
        7,
        &None,
        "".to_string(),
    )
    .unwrap();
    // `0.0000001` is NOT within 7 places: round(1e-7, 7) == 1e-7 != 0 -> fail.
    assert_almost_eq(&PyValue::from(1.0), &PyValue::from(1.0000001), 7, &None, "".to_string())
        .err()
        .unwrap();
    // At 5 places the same diff passes (round(1e-7, 5) == 0).
    assert_almost_eq(&PyValue::from(1.0), &PyValue::from(1.0000001), 5, &None, "".to_string())
        .unwrap();
    // `delta=` tolerance: abs(1.01 - 1.0) == 0.01 <= 0.1 -> pass.
    assert_almost_eq(
        &PyValue::from(1.0),
        &PyValue::from(1.01),
        7,
        &Some::<PyValue>(PyValue::from(0.1)),
        "".to_string(),
    )
    .unwrap();
    // An int delta is coerced: abs(2 - 1) == 1 <= 1 -> pass.
    assert_almost_eq(
        &PyValue::from(1i64),
        &PyValue::from(2i64),
        7,
        &Some::<PyValue>(PyValue::from(1i64)),
        "".to_string(),
    )
    .unwrap();
    // 1 vs 5 differs well beyond 7 places -> fail.
    assert_almost_eq(&PyValue::from(1i64), &PyValue::from(5i64), 7, &None, "".to_string())
        .err()
        .unwrap();
    // assertNotAlmostEqual is the negation.
    assert_not_almost_eq(
        &PyValue::from(1.0),
        &PyValue::from(1.0000001),
        7,
        &None,
        "".to_string(),
    )
    .unwrap();
    assert_not_almost_eq(&PyValue::from(1.0), &PyValue::from(1.0), 7, &None, "".to_string())
        .err()
        .unwrap();
}

#[test]
fn unittest_assert_order_comparisons_match_cpython() {
    // Verified against CPython 3.14.1 (`assertGreater`/`assertGreaterEqual`/
    // `assertLess`/`assertLessEqual`) for int/float/bool numeric ordering and
    // str lexicographic ordering.
    use stdpython::stdlib::unittest::{
        assert_greater,
        assert_greater_equal,
        assert_less,
        assert_less_equal,
    };
    use stdpython::PyValue;

    assert_greater(&PyValue::from(2i64), &PyValue::from(1i64), "".to_string()).unwrap();
    assert_greater(&PyValue::from(2.5), &PyValue::from(2), "".to_string()).unwrap();
    // bool orders numerically (CPython: 2 > True, since True == 1).
    assert_greater(&PyValue::from(2i64), &PyValue::from(true), "".to_string()).unwrap();
    assert_greater_equal(&PyValue::from(2i64), &PyValue::from(2i64), "".to_string()).unwrap();
    assert_less(&PyValue::from(1i64), &PyValue::from(2i64), "".to_string()).unwrap();
    assert_less(&PyValue::from("a".to_string()), &PyValue::from("b".to_string()), "".to_string())
        .unwrap();
    assert_less_equal(&PyValue::from(1i64), &PyValue::from(1i64), "".to_string()).unwrap();

    // Failures carry the ordering phrase.
    assert_eq!(
        assert_greater(&PyValue::from(1i64), &PyValue::from(2i64), "".to_string())
            .err()
            .unwrap()
            .message,
        "1 not greater than 2"
    );
    assert_eq!(
        assert_greater_equal(&PyValue::from(1i64), &PyValue::from(2i64), "".to_string())
            .err()
            .unwrap()
            .message,
        "1 not greater than or equal to 2"
    );
    assert_eq!(
        assert_less(&PyValue::from(2i64), &PyValue::from(1i64), "".to_string())
            .err()
            .unwrap()
            .message,
        "2 not less than 1"
    );
    // The msg= context prefixes the message.
    assert_eq!(
        assert_less_equal(&PyValue::from(2i64), &PyValue::from(1i64), "ctx".to_string())
            .err()
            .unwrap()
            .message,
        "ctx: 2 not less than or equal to 1"
    );
    // Incomparable operands fail loudly rather than silently guessing.
    assert!(
        assert_greater(&PyValue::from("a".to_string()), &PyValue::from(1i64), "".to_string())
            .err()
            .unwrap()
            .matches("AssertionError")
    );
}

#[test]
fn a_boxed_complex_behaves_like_cpythons_complex() {
    // issue #366: `[1, 2.5, 3j, "x"]` carries its complex member boxed.
    let z = PyValue::from(Complex::new(0.0, 3.0));
    // str([1, 2.5, 3j, 'x'])  == "[1, 2.5, 3j, 'x']"
    assert_eq!(py_display(&z), "3j");
    // type(3j).__name__ == 'complex'
    assert_eq!(py_value_type_name(&z), "complex");
    // bool(0j) is False, bool(3j) is True
    assert!(z.is_truthy());
    assert!(!PyValue::from(Complex::new(0.0, 0.0)).is_truthy());
    // 1j * 1j == -1 (a complex with zero imaginary part equals the real);
    // 1 + 1j != 1; 2j != 3j
    let minus_one = PyValue::from(Complex::new(-1.0, 0.0));
    assert!(stdpython::stdlib::unittest::assert_eq(&minus_one, &PyValue::from(-1i64), String::new()).is_ok());
    assert!(stdpython::stdlib::unittest::assert_eq(&PyValue::from(Complex::new(2.5, 0.0)), &PyValue::from(2.5f64), String::new()).is_ok());
    assert!(stdpython::stdlib::unittest::assert_not_eq(&PyValue::from(Complex::new(1.0, 1.0)), &PyValue::from(1i64), String::new()).is_ok());
    let err = stdpython::stdlib::unittest::assert_eq(&PyValue::from(Complex::new(0.0, 1.0)), &PyValue::from(1i64), String::new()).unwrap_err();
    // AssertionError: 1j != 1
    assert_eq!(err.message, "1j != 1");
}

#[test]
fn boxed_values_subtract_multiply_and_compare_like_cpython() {
    // issue #356: a `threading.local()` attribute is boxed, so its uses
    // go through PyValue's operators.
    // 3.5 - 1 == 2.5 ; 2 * 2.5 == 5.0 ; 1j * 2 == 2j
    assert_eq!(PyValue::Float(3.5).py_sub(&1i64), PyValue::Float(2.5));
    assert_eq!(PyValue::Int(2).py_mul(&2.5f64), PyValue::Float(5.0));
    assert_eq!(py_display(&PyValue::from(Complex::new(0.0, 1.0)).py_mul(&2i64)), "2j");
    // 'ab' * 2 == 'abab' ; b'x' * 3 == b'xxx' ; (1, 2) * 2 == (1, 2, 1, 2)
    assert_eq!(PyValue::from("ab").py_mul(&2i64), PyValue::from("abab"));
    assert_eq!(py_display(&PyValue::Bytes(b"x".to_vec()).py_mul(&3i64)), "b'xxx'");
    assert_eq!(
        py_display(&PyValue::from(vec![PyValue::Int(1), PyValue::Int(2)]).py_mul(&2i64)),
        "(1, 2, 1, 2)"
    );
    // 'x' == 'x' against a boxed str; 3 == 3.0 across the tower
    assert!(String::from("x") == PyValue::from("x"));
    assert!(PyValue::Float(3.0) == 3i64);
    // f'{7:04x}' == '0007' ; f'{True:x}' == '1' ; f'{"s":x}' is a ValueError
    assert_eq!(py_value_format_int(&PyValue::Int(7), 'x').unwrap(), 7);
    assert_eq!(py_value_format_int(&PyValue::Bool(true), 'x').unwrap(), 1);
    let err = py_value_format_int(&PyValue::from("s"), 'x').unwrap_err();
    assert_eq!(err.message, "Unknown format code 'x' for object of type 'str'");
}

#[test]
#[should_panic(expected = "can't multiply sequence by non-int of type 'float'")]
fn a_boxed_sequence_times_a_float_is_cpythons_type_error() {
    // 'a' * 2.0 -> TypeError: can't multiply sequence by non-int of type 'float'
    let _ = PyValue::from("a").py_mul(&2.0f64);
}

#[test]
fn a_list_of_numbers_boxes_and_unboxes() {
    // issue #334: `self.assertEqual(intranges_from_list([111]), (x,))` —
    // an unknown-typed list[int] result boxes like the str list does.
    assert_eq!(py_display(&PyValue::from(vec![1i64, 2])), "(1, 2)");
    assert_eq!(py_display(&PyValue::from(vec![0.5f64])), "(0.5,)");
    // A boxed int sequence into a list[int] slot; True counts as 1.
    let v: Vec<i64> = PyValue::from(vec![PyValue::Int(3), PyValue::Bool(true)]).into();
    assert_eq!(v, vec![3, 1]);
}

#[test]
fn a_boxed_range_is_still_a_range() {
    // issue #335: a range passed into a boxed parameter slot.
    let r = PyValue::from(range_start_stop(3, 6));
    // repr(range(3, 6)) == 'range(3, 6)'; range(0, 10, 2) keeps its step
    assert_eq!(py_display(&r), "range(3, 6)");
    assert_eq!(py_display(&PyValue::from(range(2))), "range(0, 2)");
    assert_eq!(py_display(&PyValue::from(range_start_stop_step(0, 10, 2).unwrap())), "range(0, 10, 2)");
    // len(range(0, 10, 3)) == 4; range(0, 10, 3)[2] == 6; 3 in range(0, 10, 3)
    let r3 = PyValue::from(range_start_stop_step(0, 10, 3).unwrap());
    assert_eq!(r3.len(), 4);
    assert_eq!(r3.py_index(2i64).unwrap(), PyValue::Int(6));
    assert!(r3.py_contains(&PyValue::Int(3)));
    // range(3)[5] -> IndexError: range object index out of range
    assert_eq!(
        PyValue::from(range(3)).py_index(5i64).unwrap_err().message,
        "range object index out of range"
    );
    // bool(range(0)) is False; list(range(3, 6)) == [3, 4, 5]
    assert!(!PyValue::from(range(0)).is_truthy());
    let v: Vec<i64> = r.clone().into();
    assert_eq!(v, vec![3, 4, 5]);
    // range(0, 3, 2) == range(0, 4, 2); range(0) == range(2, 2)
    assert!(stdpython::stdlib::unittest::assert_eq(
        &PyValue::from(range_start_stop_step(0, 3, 2).unwrap()),
        &PyValue::from(range_start_stop_step(0, 4, 2).unwrap()),
        String::new()
    )
    .is_ok());
    assert!(stdpython::stdlib::unittest::assert_eq(
        &PyValue::from(range(0)),
        &PyValue::from(range_start_stop(2, 2)),
        String::new()
    )
    .is_ok());
}

#[test]
fn a_variable_length_tuple_prints_as_a_tuple() {
    // issue #399: the runtime variable-length tuple.
    let t = PyTuple(vec![3i64, 4]);
    // repr((3, 4)) == '(3, 4)'; str((7,)) == '(7,)'; repr(()) == '()'
    assert_eq!(py_display(&t), "(3, 4)");
    assert_eq!(py_display(&PyTuple(vec![7i64])), "(7,)");
    assert_eq!(py_display(&PyTuple::<i64>(vec![])), "()");
    // repr(('b', 'c')) == "('b', 'c')"
    assert_eq!(
        py_display(&PyTuple(vec!["b".to_string(), "c".to_string()])),
        "('b', 'c')"
    );
    // (3, 4)[1:] == (4,); (3, 4) + (5,) == (3, 4, 5); (1, 2) * 2 == (1, 2, 1, 2)
    assert_eq!(py_display(&t.py_slice(Some(1), None, None)), "(4,)");
    assert_eq!(py_display(&t.py_add(&PyTuple(vec![5]))), "(3, 4, 5)");
    assert_eq!(py_display(&PyTuple(vec![1i64, 2]).py_mul(&2)), "(1, 2, 1, 2)");
    // 3 in (3, 4); len(()) == 0; bool(()) is False; sum((1, 2, 3)) == 6
    assert!(t.py_contains(&3));
    assert_eq!(PyTuple::<i64>(vec![]).len(), 0);
    assert!(!PyTuple::<i64>(vec![]).is_truthy());
    assert_eq!(sum(PyTuple(vec![1i64, 2, 3])), 6);
    // (3, 4)[5] -> IndexError: tuple index out of range
    assert_eq!(t.py_index(5i64).unwrap_err().message, "tuple index out of range");
    // (3, 4)[-1] == 4; list((3, 4)) == [3, 4]
    assert_eq!(t.py_index(-1i64).unwrap(), 4);
    assert_eq!(list(t.clone()), vec![3, 4]);
    // A boxed tuple is still a tuple: repr(tuple([1, 2])) == '(1, 2)'
    assert_eq!(py_display(&PyValue::from(t)), "(3, 4)");
}

#[test]
fn fixed_tuples_of_every_arity_print_as_tuples() {
    // repr((7,)) == '(7,)'
    assert_eq!(py_display(&(7i64,)), "(7,)");
    // repr((1, 'a', 2.5, None)) == "(1, 'a', 2.5, None)"
    assert_eq!(
        py_display(&(1i64, "a".to_string(), 2.5f64, None::<i64>)),
        "(1, 'a', 2.5, None)"
    );
}

#[test]
fn enumerate_takes_strings_and_boxed_values() {
    // list(enumerate('ab')) == [(0, 'a'), (1, 'b')]
    assert_eq!(py_display(&enumerate("ab")), "[(0, 'a'), (1, 'b')]");
    // list(enumerate('xy', 1)) == [(1, 'x'), (2, 'y')]
    assert_eq!(py_display(&enumerate_start("xy".to_string(), 1)), "[(1, 'x'), (2, 'y')]");
    // list(enumerate(b'ab')) == [(0, 97), (1, 98)]
    assert_eq!(
        py_display(&enumerate(PyValue::from(b"ab".to_vec()))),
        "[(0, 97), (1, 98)]"
    );
    // list(enumerate(('x', 1))) == [(0, 'x'), (1, 1)]
    let t = PyValue::from(PyTuple(vec![PyValue::from("x"), PyValue::from(1i64)]));
    assert_eq!(py_display(&enumerate(t)), "[(0, 'x'), (1, 1)]");
}

#[test]
fn boxed_values_order_like_cpython() {
    let t = |v: Vec<PyValue>| PyValue::from(PyTuple(v));
    let (i, s) = (|x: i64| PyValue::from(x), |x: &str| PyValue::from(x));
    // (1, 'a') < (1, 'b'); (1,) < (1, 'a'); (2, 'Z') < (10, '3')
    assert!(t(vec![i(1), s("a")]).py_lt_boxed(&t(vec![i(1), s("b")])).unwrap());
    assert!(t(vec![i(1)]).py_lt_boxed(&t(vec![i(1), s("a")])).unwrap());
    assert!(t(vec![i(2), s("Z")]).py_lt_boxed(&t(vec![i(10), s("3")])).unwrap());
    // 1 < 2.5; True < 2; float('nan') < 1 is False; b'a' < b'b'
    assert!(i(1).py_lt_boxed(&PyValue::from(2.5)).unwrap());
    assert!(PyValue::from(true).py_lt_boxed(&i(2)).unwrap());
    assert!(!PyValue::from(f64::NAN).py_lt_boxed(&i(1)).unwrap());
    assert!(PyValue::from(b"a".to_vec()).py_lt_boxed(&PyValue::from(b"b".to_vec())).unwrap());
    // (1, 'a') < ('x',) -> TypeError: '<' not supported between instances of 'int' and 'str'
    let err = t(vec![i(1), s("a")]).py_lt_boxed(&t(vec![s("x")])).unwrap_err();
    assert_eq!(err.message, "'<' not supported between instances of 'int' and 'str'");
    // bisect.bisect_left(((0, '3'), (65, 'M', 'a'), (91, 'V')), (66, 'Z')) == 2
    let table = t(vec![
        t(vec![i(0), s("3")]),
        t(vec![i(65), s("M"), s("a")]),
        t(vec![i(91), s("V")]),
    ]);
    let probe = t(vec![i(66), s("Z")]);
    assert_eq!(stdpython::stdlib::bisect::bisect_left(&table, &probe, 0, None).unwrap(), 2);
    // bisect.bisect_right(table, (91, 'V')) == 3
    let probe = t(vec![i(91), s("V")]);
    assert_eq!(stdpython::stdlib::bisect::bisect_right(&table, &probe, 0, None).unwrap(), 3);
}

#[test]
fn unpacking_a_sequence_checks_its_length() {
    // a, b = [1, 2] -> 1, 2
    assert_eq!(unpack_sequence::<i64, 2>(vec![1, 2]).unwrap(), [1, 2]);
    // a, b = [1, 2, 3] -> ValueError: too many values to unpack (expected 2)
    assert_eq!(
        unpack_sequence::<i64, 2>(vec![1, 2, 3]).unwrap_err().message,
        "too many values to unpack (expected 2)"
    );
    // a, b = [1] -> ValueError: not enough values to unpack (expected 2, got 1)
    assert_eq!(
        unpack_sequence::<i64, 2>(PyTuple(vec![1])).unwrap_err().message,
        "not enough values to unpack (expected 2, got 1)"
    );
}

#[test]
fn boxed_argument_binding_raises_cpythons_type_errors() {
    let params = ["s", "upper"];
    let s = || PyValue::from("a");
    // def enc(s, upper=False): enc() -> missing 1 required positional argument: 's'
    assert_eq!(
        bind_boxed_args("enc", &params, 1, (vec![], vec![])).unwrap_err().message,
        "enc() missing 1 required positional argument: 's'"
    );
    // enc("a", True, 3) -> takes from 1 to 2 positional arguments but 3 were given
    assert_eq!(
        bind_boxed_args("enc", &params, 1, (vec![s(), s(), s()], vec![]))
            .unwrap_err()
            .message,
        "enc() takes from 1 to 2 positional arguments but 3 were given"
    );
    // enc("a", colour=1) -> got an unexpected keyword argument 'colour'
    assert_eq!(
        bind_boxed_args("enc", &params, 1, (vec![s()], vec![("colour".into(), s())]))
            .unwrap_err()
            .message,
        "enc() got an unexpected keyword argument 'colour'"
    );
    // enc("a", s="b") -> got multiple values for argument 's'
    assert_eq!(
        bind_boxed_args("enc", &params, 1, (vec![s()], vec![("s".into(), s())]))
            .unwrap_err()
            .message,
        "enc() got multiple values for argument 's'"
    );
    // enc(upper=True, s="a") binds both by name
    let bound =
        bind_boxed_args("enc", &params, 1, (vec![], vec![("upper".into(), s()), ("s".into(), s())]))
            .unwrap();
    assert!(bound.iter().all(Option::is_some));
    // None(1) -> TypeError: 'NoneType' object is not callable
    assert_eq!(
        PyValue::None_.call_boxed(vec![s()]).unwrap_err().message,
        "'NoneType' object is not callable"
    );
}

#[test]
fn boxed_call_spreads_raise_cpythons_type_errors() {
    // def f(*a, **k): return len(a) + len(k)   (the entry module's f)
    let f = PyValue::Function(stdpython::PyCallable::function(
        "f",
        "__main__.f",
        |(a, k): stdpython::BoxedArgs| Ok(PyValue::Int((a.len() + k.len()) as i64)),
    ));
    let mut args = vec![PyValue::Int(1)];
    // f(1, *(2, 3)) -> the members append in order
    f.extend_spread_args(&mut args, PyValue::from((2i64, 3i64))).unwrap();
    assert_eq!(args.len(), 3);
    // f(*5) -> __main__.f() argument after * must be an iterable, not int
    assert_eq!(
        f.extend_spread_args(&mut args, PyValue::Int(5)).unwrap_err().message,
        "__main__.f() argument after * must be an iterable, not int"
    );
    // f(**5) -> __main__.f() argument after ** must be a mapping, not int
    let mut kw: Vec<(String, PyValue)> = vec![("a".to_string(), PyValue::Int(1))];
    assert_eq!(
        f.extend_spread_kwargs(&mut kw, PyValue::Int(5)).unwrap_err().message,
        "__main__.f() argument after ** must be a mapping, not int"
    );
    // f(a=1, **{"a": 2}) -> __main__.f() got multiple values for keyword argument 'a'
    let mut d: stdpython::PyDict<String, PyValue> = stdpython::PyDict::new();
    d.insert("a".to_string(), PyValue::Int(2));
    assert_eq!(
        f.extend_spread_kwargs(&mut kw, PyValue::from(d)).unwrap_err().message,
        "__main__.f() got multiple values for keyword argument 'a'"
    );
}

#[test]
fn boxed_and_list_unpacking_raise_cpythons_errors() {
    use stdpython::{unpack_boxed, PyUnpack};
    // a, b = ("x", 1) boxed -> 'x', 1
    let [a, b] = unpack_boxed::<2>(PyValue::from(("x", 1i64))).unwrap();
    assert_eq!((a, b), (PyValue::from("x"), PyValue::Int(1)));
    // a, b = "ab" -> 'a', 'b' (a str iterates its characters)
    let [a, b] = unpack_boxed::<2>(PyValue::from("ab")).unwrap();
    assert_eq!((a, b), (PyValue::from("a"), PyValue::from("b")));
    // a, b = 5 -> TypeError: cannot unpack non-iterable int object
    let e = unpack_boxed::<2>(PyValue::Int(5)).unwrap_err();
    assert_eq!((e.exception_type.as_str(), e.message.as_str()), ("TypeError", "cannot unpack non-iterable int object"));
    // a, b = None -> TypeError: cannot unpack non-iterable NoneType object
    let e = unpack_boxed::<2>(PyValue::None_).unwrap_err();
    assert_eq!(e.message, "cannot unpack non-iterable NoneType object");
    // a, b = "abc" -> ValueError: too many values to unpack (expected 2)
    let e = unpack_boxed::<2>(PyValue::from("abc")).unwrap_err();
    assert_eq!((e.exception_type.as_str(), e.message.as_str()), ("ValueError", "too many values to unpack (expected 2)"));
    // a, b = (1, 2) -> the tuple itself
    assert_eq!(PyUnpack::<2>::py_unpack((1i64, "s")).unwrap(), (1, "s"));
    // a, b = [7] -> ValueError: not enough values to unpack (expected 2, got 1)
    let e = PyUnpack::<2>::py_unpack(vec![7i64]).unwrap_err();
    assert_eq!(e.message, "not enough values to unpack (expected 2, got 1)");
    // a, b = [7, 8] -> 7, 8
    assert_eq!(PyUnpack::<2>::py_unpack(vec![7i64, 8]).unwrap(), (7, 8));
}

#[test]
fn py_class_name_reads_the_python_class_name_off_the_type() {
    mod m {
        pub struct Session;
        pub struct Self_;
        // A Python class named `type`, as the generated crate spells it.
        #[allow(non_camel_case_types)]
        pub struct r#type;
        pub struct G<T>(pub T);
    }
    // type(Session()).__name__ -> 'Session'
    assert_eq!(stdpython::py_class_name::<m::Session>(), "Session");
    // a class named Self (renamed Self_ in Rust) -> 'Self'
    assert_eq!(stdpython::py_class_name::<m::Self_>(), "Self");
    // a class named type (a raw identifier) -> 'type'
    assert_eq!(stdpython::py_class_name::<m::r#type>(), "type");
    // generics never leak into the name
    assert_eq!(stdpython::py_class_name::<m::G<m::Session>>(), "G");
}

#[test]
fn str_of_an_optional_value_matches_cpython() {
    use stdpython::str;
    // Verified against python3:
    //   str(None) == 'None'; x: int | None = 5; str(x) == '5'
    //   str(True) == 'True'
    assert_eq!(str(None::<i64>), "None");
    assert_eq!(str(Some(5i64)), "5");
    assert_eq!(str(Some(true)), "True");
    assert_eq!(str(Some("a".to_string())), "a");
}

// ---------------------------------------------------------------------------
// collections.deque / defaultdict / OrderedDict (issue #427). Every expected
// value below is the output of the Python expression in the comment, run
// under python3.12.
// ---------------------------------------------------------------------------
mod collections_containers {
    use stdpython::collections::{defaultdict, deque, OrderedDict};
    use stdpython::*;

    fn items<T: Clone>(d: &deque<T>) -> Vec<T> {
        d.to_vec()
    }

    #[test]
    fn deque_end_operations_and_empty_pops() {
        let mut d = deque::from_iter(vec![1i64, 2, 3], None);
        // d = deque([1,2,3]); d.append(4); d.appendleft(0) -> deque([0, 1, 2, 3, 4])
        d.append(4);
        d.appendleft(0);
        assert_eq!(items(&d), vec![0, 1, 2, 3, 4]);
        // d.pop() -> 4; d.popleft() -> 0; d -> deque([1, 2, 3])
        assert_eq!(d.pop().unwrap(), 4);
        assert_eq!(d.popleft().unwrap(), 0);
        assert_eq!(items(&d), vec![1, 2, 3]);
        // deque().pop() / deque().popleft() -> IndexError: pop from an empty deque
        let mut e: deque<i64> = deque::new();
        for err in [e.pop().unwrap_err(), e.popleft().unwrap_err()] {
            assert_eq!(err.exception_type, "IndexError");
            assert_eq!(err.message, "pop from an empty deque");
        }
    }

    #[test]
    fn deque_maxlen_trims_the_opposite_end() {
        let mut m = deque::construct(vec![1i64, 2, 3], Some(3)).unwrap();
        // m = deque([1,2,3], maxlen=3); m.append(4) -> deque([2, 3, 4], maxlen=3)
        m.append(4);
        assert_eq!(items(&m), vec![2, 3, 4]);
        // m.appendleft(0) -> deque([0, 2, 3], maxlen=3)
        m.appendleft(0);
        assert_eq!(items(&m), vec![0, 2, 3]);
        // m.extend([7, 8]) -> deque([3, 7, 8], maxlen=3)
        m.extend(vec![7, 8]);
        assert_eq!(items(&m), vec![3, 7, 8]);
        // m.extendleft([5, 6]) -> deque([6, 5, 3], maxlen=3)
        m.extendleft(vec![5, 6]);
        assert_eq!(items(&m), vec![6, 5, 3]);
        // m.maxlen -> 3; deque([1]).maxlen -> None
        assert_eq!(m.maxlen(), Some(3));
        assert_eq!(deque::<i64>::new().maxlen(), None);
        // deque([1,2,3], maxlen=0) -> deque([], maxlen=0)
        let z = deque::construct(vec![1i64, 2, 3], Some(0)).unwrap();
        assert_eq!(z.len(), 0);
        // deque([1], maxlen=-1) -> ValueError: maxlen must be non-negative
        let err = deque::construct(vec![1i64], Some(-1)).unwrap_err();
        assert_eq!(err.exception_type, "ValueError");
        assert_eq!(err.message, "maxlen must be non-negative");
    }

    #[test]
    fn deque_rotate_matches_python() {
        let mut r = deque::from_iter(vec![1i64, 2, 3, 4, 5], None);
        // r.rotate(2) -> deque([4, 5, 1, 2, 3])
        r.rotate(2);
        assert_eq!(items(&r), vec![4, 5, 1, 2, 3]);
        // r.rotate(-7) -> deque([1, 2, 3, 4, 5])
        r.rotate(-7);
        assert_eq!(items(&r), vec![1, 2, 3, 4, 5]);
        // r.rotate() (default 1) -> deque([5, 1, 2, 3, 4]); r.rotate(0) leaves it
        r.rotate(1);
        r.rotate(0);
        assert_eq!(items(&r), vec![5, 1, 2, 3, 4]);
        // deque().rotate(3) -> deque([])
        let mut e: deque<i64> = deque::new();
        e.rotate(3);
        assert_eq!(e.len(), 0);
        // x = deque([1,2,3]); x.extendleft([7,8,9]) -> deque([9, 8, 7, 1, 2, 3])
        let mut x = deque::from_iter(vec![1i64, 2, 3], None);
        x.extendleft(vec![7, 8, 9]);
        assert_eq!(items(&x), vec![9, 8, 7, 1, 2, 3]);
    }

    #[test]
    fn deque_repr_matches_python() {
        // repr(deque(['a', 'b'])) -> "deque(['a', 'b'])"
        let s = deque::from_iter(vec!["a".to_string(), "b".to_string()], None);
        assert_eq!(s.py_repr(), "deque(['a', 'b'])");
        // repr(deque()) -> 'deque([])'
        assert_eq!(deque::<i64>::new().py_repr(), "deque([])");
        // repr(deque([1], maxlen=2)) -> 'deque([1], maxlen=2)'
        assert_eq!(
            deque::from_iter(vec![1i64], Some(2)).py_repr(),
            "deque([1], maxlen=2)"
        );
    }

    #[test]
    fn deque_indexing_remove_and_equality() {
        let mut y = deque::from_iter(vec![1i64, 2, 3], None);
        // y[0], y[-1] -> 1, 3
        assert_eq!((y.py_index(0).unwrap(), y.py_index(-1).unwrap()), (1, 3));
        // y[3] / y[-4] -> IndexError: deque index out of range
        for i in [3, -4] {
            let err = y.py_index(i).unwrap_err();
            assert_eq!(err.exception_type, "IndexError");
            assert_eq!(err.message, "deque index out of range");
        }
        // y[1] = 20 -> deque([1, 20, 3]); y[9] = 1 -> IndexError: deque index out of range
        y.py_set_index(1, 20).unwrap();
        assert_eq!(items(&y), vec![1, 20, 3]);
        assert_eq!(
            y.py_set_index(9, 1).unwrap_err().message,
            "deque index out of range"
        );
        // y.remove(20) -> deque([1, 3]); y.remove(99) -> ValueError: 99 is not in deque
        y.remove(&20).unwrap();
        assert_eq!(items(&y), vec![1, 3]);
        let err = y.remove(&99).unwrap_err();
        assert_eq!(err.exception_type, "ValueError");
        assert_eq!(err.message, "99 is not in deque");
        // deque(["a"]).remove("zz") -> ValueError: 'zz' is not in deque
        let mut s = deque::from_iter(vec!["a".to_string()], None);
        assert_eq!(
            s.remove(&"zz".to_string()).unwrap_err().message,
            "'zz' is not in deque"
        );
        // deque([1, 2]) == deque([1, 2]) -> True; == deque([2, 1]) -> False;
        // deque([1], maxlen=3) == deque([1]) -> True (the bound is not compared)
        assert!(deque::from_iter(vec![1i64, 2], None) == deque::from_iter(vec![1i64, 2], None));
        assert!(deque::from_iter(vec![1i64, 2], None) != deque::from_iter(vec![2i64, 1], None));
        assert!(deque::from_iter(vec![1i64], Some(3)) == deque::from_iter(vec![1i64], None));
        // 2 in deque([1, 2]) -> True; 5 in ... -> False
        let c = deque::from_iter(vec![1i64, 2], None);
        assert!(c.py_contains(&2) && !c.py_contains(&5));
    }

    #[test]
    fn deque_del_item_removes_by_index() {
        let mut d = deque::from_iter(vec![1i64, 2, 3], None);
        // del d[1] -> deque([1, 3]); del d[-1] -> deque([1])
        assert_eq!(d.py_pop(1).unwrap(), 2);
        assert_eq!(items(&d), vec![1, 3]);
        assert_eq!(d.py_pop(-1).unwrap(), 3);
        assert_eq!(items(&d), vec![1]);
        // del d[5] -> IndexError: deque index out of range
        let err = d.py_pop(5).unwrap_err();
        assert_eq!(err.exception_type, "IndexError");
        assert_eq!(err.message, "deque index out of range");
    }

    #[test]
    fn defaultdict_read_inserts_and_get_does_not() {
        let dd: defaultdict<String, i64> = defaultdict::with_class(|| 0i64, "int");
        // dd["a"] -> 0, and the key is now present: len(dd) -> 1, list(dd) -> ['a']
        assert_eq!(dd.py_index("a").unwrap(), 0);
        assert_eq!(dd.len(), 1);
        assert_eq!(dd.py_keys(), vec!["a".to_string()]);
        // dd.get("zz") -> None and does NOT insert; "zz" in dd -> False
        assert_eq!(dd.py_get(&"zz".to_string()), None);
        assert!(!dd.py_contains("zz"));
        assert_eq!(dd.len(), 1);
    }

    #[test]
    fn defaultdict_repr_order_and_missing_keys() {
        let mut dd: defaultdict<String, i64> = defaultdict::with_class(|| 0i64, "int");
        // dd["a"]; dd["b"] += 2 -> defaultdict(<class 'int'>, {'a': 0, 'b': 2})
        let _ = dd.py_index("a").unwrap();
        *dd.py_index_mut("b").unwrap() += 2;
        assert_eq!(dd.py_repr(), "defaultdict(<class 'int'>, {'a': 0, 'b': 2})");
        // list(dd.keys()), list(dd.values()) -> ['a', 'b'], [0, 2]
        assert_eq!(dd.py_values(), vec![0, 2]);
        // dd.pop("a") -> 0; dd.pop("zz") -> KeyError: 'zz'; dd.pop("zz", -1) -> -1
        assert_eq!(dd.py_pop("a".to_string()).unwrap(), 0);
        assert_eq!(dd.py_pop("zz".to_string()).unwrap_err().message, "'zz'");
        assert_eq!(dd.py_pop_default("zz".to_string(), -1), -1);
        // defaultdict()["k"] / defaultdict(None)["k"] -> KeyError: 'k'
        let plain: defaultdict<String, i64> = defaultdict::without_factory();
        let err = plain.py_index("k").unwrap_err();
        assert_eq!((err.exception_type.as_str(), err.message.as_str()), ("KeyError", "'k'"));
        // repr(defaultdict(list)) -> "defaultdict(<class 'list'>, {})"
        let l: defaultdict<String, Vec<i64>> = defaultdict::with_class(|| Vec::new(), "list");
        assert_eq!(l.py_repr(), "defaultdict(<class 'list'>, {})");
        // dl["k"].append(1); dl["k"].append(2); dl["j"] -> {'k': [1, 2], 'j': []}
        let mut l = l;
        l.py_index_mut("k").unwrap().push(1);
        l.py_index_mut("k").unwrap().push(2);
        let _ = l.py_index("j").unwrap();
        assert_eq!(l.py_repr(), "defaultdict(<class 'list'>, {'k': [1, 2], 'j': []})");
        // defaultdict(float, {"x": 1.5}) -> defaultdict(<class 'float'>, {'x': 1.5})
        let mut start = PyDict::default();
        start.insert("x".to_string(), 1.5f64);
        let f: defaultdict<String, f64> =
            defaultdict::with_class(|| 0.0f64, "float").with_items(start);
        assert_eq!(f.py_repr(), "defaultdict(<class 'float'>, {'x': 1.5})");
    }

    #[test]
    fn defaultdict_equality_ignores_the_factory() {
        let mut start = PyDict::default();
        start.insert("a".to_string(), 1i64);
        let a: defaultdict<String, i64> =
            defaultdict::with_class(|| 0i64, "int").with_items(start.clone());
        let b: defaultdict<String, i64> = defaultdict::new(|| 5i64).with_items(start.clone());
        // defaultdict(int, {"a": 1}) == defaultdict(lambda: 5, {"a": 1}) -> True
        assert!(a == b);
        // defaultdict(int, {"a": 1}) == {"a": 1} -> True
        assert!(a == start);
    }

    #[test]
    fn a_factoryless_defaultdict_raises_a_genuine_keyerror() {
        // defaultdict()["k"] / defaultdict(None)["k"] -> KeyError: 'k' — a
        // program may catch it with `except KeyError`.
        let d = defaultdict::<String, i64>::without_factory();
        let err = d.py_index("k").unwrap_err();
        assert_eq!(err.exception_type, "KeyError");
        assert_eq!(err.message, "'k'");
    }

    #[test]
    #[should_panic(expected = "built without its default_factory")]
    fn a_default_built_defaultdict_panics_on_a_missing_key_read() {
        // `Default::default()` is a class struct's placeholder, NOT a
        // `defaultdict()`: a KeyError here would be mistaken for CPython's
        // by `except KeyError:`, so a missing-key read panics naming the bug.
        let d: defaultdict<String, i64> = Default::default();
        let _ = d.py_index("k");
    }

    #[test]
    #[should_panic(expected = "built without its default_factory")]
    fn a_default_built_defaultdict_panics_on_a_missing_key_slot() {
        let mut d: defaultdict<String, Vec<i64>> = Default::default();
        let _ = d.py_index_mut("k");
    }

    #[test]
    fn a_default_built_defaultdict_still_serves_present_keys_and_reads_that_never_insert() {
        let mut d: defaultdict<String, i64> = Default::default();
        d.py_set_index("a".to_string(), 1).unwrap();
        assert_eq!(d.py_index("a").unwrap(), 1);
        assert_eq!(d.py_get(&"zz".to_string()), None);
        assert!(!d.py_contains("zz"));
    }

    #[test]
    #[should_panic(expected = "memory address")]
    fn defaultdict_repr_with_a_lambda_factory_is_loud() {
        // repr(defaultdict(lambda: 5)) prints `<function <lambda> at 0x...>`: an
        // address no run reproduces, so rython panics instead of printing a
        // different string.
        let d: defaultdict<String, i64> = defaultdict::new(|| 5);
        let _ = d.py_repr();
    }

    #[test]
    fn ordereddict_order_move_to_end_and_popitem() {
        let mut od: OrderedDict<String, i64> = OrderedDict::new();
        for (k, v) in [("b", 1), ("a", 2), ("c", 3)] {
            od.py_set_index(k.to_string(), v).unwrap();
        }
        // repr(od) -> "OrderedDict({'b': 1, 'a': 2, 'c': 3})" (3.12 form); repr(OrderedDict()) -> 'OrderedDict()'
        assert_eq!(od.py_repr(), "OrderedDict({'b': 1, 'a': 2, 'c': 3})");
        assert_eq!(OrderedDict::<String, i64>::new().py_repr(), "OrderedDict()");
        // od.move_to_end("b") -> ['a', 'c', 'b']
        od.move_to_end(&"b".to_string(), true).unwrap();
        assert_eq!(od.keys(), vec!["a", "c", "b"]);
        // od.move_to_end("c", last=False) -> ['c', 'a', 'b']
        od.move_to_end(&"c".to_string(), false).unwrap();
        assert_eq!(od.keys(), vec!["c", "a", "b"]);
        // od.popitem() -> ('b', 1); od.popitem(last=False) -> ('c', 3); list(od) -> ['a']
        assert_eq!(od.popitem(true).unwrap(), ("b".to_string(), 1));
        assert_eq!(od.popitem(false).unwrap(), ("c".to_string(), 3));
        assert_eq!(od.keys(), vec!["a"]);
        // overwriting keeps the position; delete + reinsert goes last
        let mut o2: OrderedDict<String, i64> = OrderedDict::new();
        for (k, v) in [("a", 1), ("b", 2), ("c", 3)] {
            o2.insert(k.to_string(), v);
        }
        // o2["a"] = 10 -> [('a', 10), ('b', 2), ('c', 3)]
        o2.py_set_index("a".to_string(), 10).unwrap();
        assert_eq!(o2.items()[0], ("a".to_string(), 10));
        // del o2["b"]; o2["b"] = 7 -> [('a', 10), ('c', 3), ('b', 7)]
        o2.remove(&"b".to_string());
        o2.py_set_index("b".to_string(), 7).unwrap();
        assert_eq!(o2.keys(), vec!["a", "c", "b"]);
    }

    #[test]
    fn ordereddict_errors_carry_cpythons_text() {
        let mut od: OrderedDict<String, i64> = OrderedDict::new();
        // OrderedDict().popitem() -> KeyError: 'dictionary is empty'
        let err = od.popitem(true).unwrap_err();
        assert_eq!(
            (err.exception_type.as_str(), err.message.as_str()),
            ("KeyError", "'dictionary is empty'")
        );
        // OrderedDict().move_to_end("x") / ["x"] / .pop("x") -> KeyError: 'x'
        assert_eq!(od.move_to_end(&"x".to_string(), true).unwrap_err().message, "'x'");
        assert_eq!(od.py_index("x").unwrap_err().message, "'x'");
        assert_eq!(od.py_pop("x".to_string()).unwrap_err().message, "'x'");
    }

    #[test]
    fn ordereddict_equality_is_order_sensitive_only_between_ordereddicts() {
        let ab = OrderedDict::from_pairs(vec![("a".to_string(), 1i64), ("b".to_string(), 2)]);
        let ba = OrderedDict::from_pairs(vec![("b".to_string(), 2i64), ("a".to_string(), 1)]);
        // OrderedDict([("a",1),("b",2)]) == OrderedDict([("b",2),("a",1)]) -> False
        assert!(ab != ba);
        // OrderedDict([("a",1),("b",2)]) == {"b": 2, "a": 1} -> True
        let mut plain = PyDict::default();
        plain.insert("b".to_string(), 2i64);
        plain.insert("a".to_string(), 1i64);
        assert!(ab == plain);
        // repr(OrderedDict([("a", 1), ("b", 2), ("a", 3)])) -> OrderedDict({'a': 3, 'b': 2})
        let dup = OrderedDict::from_pairs(vec![
            ("a".to_string(), 1i64),
            ("b".to_string(), 2),
            ("a".to_string(), 3),
        ]);
        assert_eq!(dup.py_repr(), "OrderedDict({'a': 3, 'b': 2})");
        // repr(OrderedDict([(1, "x"), (2, "y")])) -> OrderedDict({1: 'x', 2: 'y'})
        let ints = OrderedDict::from_pairs(vec![(1i64, "x".to_string()), (2, "y".to_string())]);
        assert_eq!(ints.py_repr(), "OrderedDict({1: 'x', 2: 'y'})");
    }
}

// A BOXED OrderedDict (`PyValue::OrderedDict` — `OrderedDict(value)` over an
// untyped parameter, requests' `from_key_val_list`). Every expected value was
// captured from python3.12; the Python expression is in the comment above it.
mod boxed_ordereddict {
    use stdpython::collections::OrderedDict;
    use stdpython::*;

    fn s(v: &str) -> PyValue {
        PyValue::Str(v.to_string())
    }

    fn tup(items: Vec<PyValue>) -> PyValue {
        PyValue::Tuple(std::sync::Arc::new(items))
    }

    fn pair(k: &str, v: i64) -> PyValue {
        tup(vec![s(k), PyValue::Int(v)])
    }

    fn pairs(items: &[(&str, i64)]) -> PyValue {
        tup(items.iter().map(|(k, v)| pair(k, *v)).collect())
    }

    fn build(v: PyValue) -> PyValue {
        OrderedDict::from_boxed(v).unwrap()
    }

    fn err(v: PyValue) -> (String, String) {
        let e = OrderedDict::from_boxed(v).unwrap_err();
        (e.exception_type.clone(), e.message.clone())
    }

    fn dict(items: &[(&str, i64)]) -> PyDict<String, PyValue> {
        let mut d = PyDict::default();
        for (k, v) in items {
            d.insert(k.to_string(), PyValue::Int(*v));
        }
        d
    }

    #[test]
    fn repr_and_str_are_the_ordereddict_form() {
        // repr(OrderedDict([("b", 1), ("a", 2)])) -> OrderedDict({'b': 1, 'a': 2})
        let od = build(pairs(&[("b", 1), ("a", 2)]));
        assert_eq!(od.py_repr(), "OrderedDict({'b': 1, 'a': 2})");
        // str(OrderedDict([("a", "x")])) -> OrderedDict({'a': 'x'})
        let ax = build(tup(vec![tup(vec![s("a"), s("x")])]));
        assert_eq!(ax.py_display(), "OrderedDict({'a': 'x'})");
        // repr(OrderedDict()) -> OrderedDict()
        assert_eq!(build(tup(vec![])).py_repr(), "OrderedDict()");
        // repr(OrderedDict([("a", OrderedDict([("b", None)]))]))
        //   -> OrderedDict({'a': OrderedDict({'b': None})})
        let inner = build(tup(vec![tup(vec![s("b"), PyValue::None_])]));
        let outer = build(tup(vec![tup(vec![s("a"), inner])]));
        assert_eq!(outer.py_repr(), "OrderedDict({'a': OrderedDict({'b': None})})");
    }

    #[test]
    fn constructs_from_a_dict_and_from_pairs() {
        // OrderedDict({"x": 1, "y": 2}) -> OrderedDict({'x': 1, 'y': 2})
        let from_dict = build(PyValue::Dict(std::sync::Arc::new(dict(&[("x", 1), ("y", 2)]))));
        assert_eq!(from_dict.py_repr(), "OrderedDict({'x': 1, 'y': 2})");
        // OrderedDict([("a", 1), ("a", 2)]) -> OrderedDict({'a': 2}); a repeated key
        // keeps its first position and takes the last value.
        assert_eq!(build(pairs(&[("a", 1), ("a", 2)])).py_repr(), "OrderedDict({'a': 2})");
        // OrderedDict(["ab"]) -> OrderedDict({'a': 'b'}): a 2-character str is a pair.
        assert_eq!(build(tup(vec![s("ab")])).py_repr(), "OrderedDict({'a': 'b'})");
        // OrderedDict(OrderedDict([("a", 1)])) -> OrderedDict({'a': 1}), a copy.
        let first = build(pairs(&[("a", 1)]));
        assert_eq!(build(first).py_repr(), "OrderedDict({'a': 1})");
    }

    #[test]
    fn constructor_errors_are_cpythons() {
        // OrderedDict(5) -> TypeError: 'int' object is not iterable
        assert_eq!(
            err(PyValue::Int(5)),
            ("TypeError".to_string(), "'int' object is not iterable".to_string())
        );
        // OrderedDict(5.5) -> TypeError: 'float' object is not iterable
        assert_eq!(err(PyValue::Float(5.5)).1, "'float' object is not iterable");
        // OrderedDict(True) -> TypeError: 'bool' object is not iterable
        assert_eq!(err(PyValue::Bool(true)).1, "'bool' object is not iterable");
        // OrderedDict(None) -> TypeError: 'NoneType' object is not iterable
        assert_eq!(err(PyValue::None_).1, "'NoneType' object is not iterable");
        // OrderedDict([1]) -> TypeError: 'int' object is not iterable
        assert_eq!(
            err(tup(vec![PyValue::Int(1)])),
            ("TypeError".to_string(), "'int' object is not iterable".to_string())
        );
        // OrderedDict([(1, 2, 3)]) -> ValueError: too many values to unpack (expected 2)
        let three = tup(vec![tup(vec![PyValue::Int(1), PyValue::Int(2), PyValue::Int(3)])]);
        assert_eq!(
            err(three),
            (
                "ValueError".to_string(),
                "too many values to unpack (expected 2)".to_string()
            )
        );
        // OrderedDict([("a",)]) -> ValueError: need more than 1 value to unpack
        assert_eq!(
            err(tup(vec![tup(vec![s("a")])])),
            (
                "ValueError".to_string(),
                "need more than 1 value to unpack".to_string()
            )
        );
        // OrderedDict([()]) -> ValueError: need more than 0 values to unpack
        assert_eq!(err(tup(vec![tup(vec![])])).1, "need more than 0 values to unpack");
        // OrderedDict("ab") -> ValueError: need more than 1 value to unpack
        assert_eq!(err(s("ab")).1, "need more than 1 value to unpack");
        // OrderedDict([("a", 1), 5]) -> TypeError: 'int' object is not iterable
        assert_eq!(
            err(tup(vec![pair("a", 1), PyValue::Int(5)])).1,
            "'int' object is not iterable"
        );
        // OrderedDict([({"a": 1}, 2)]) -> TypeError: unhashable type: 'dict'
        let unhashable = tup(vec![tup(vec![
            PyValue::Dict(std::sync::Arc::new(dict(&[("a", 1)]))),
            PyValue::Int(2),
        ])]);
        assert_eq!(
            err(unhashable),
            ("TypeError".to_string(), "unhashable type: 'dict'".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "a boxed OrderedDict holds str keys only")]
    fn a_non_str_key_is_a_loud_panic_never_a_stringified_key() {
        // OrderedDict([(1, 2)]) -> OrderedDict({1: 2}) in CPython; the boxed
        // OrderedDict holds str keys only, so rython refuses loudly.
        let _ = OrderedDict::from_boxed(tup(vec![tup(vec![PyValue::Int(1), PyValue::Int(2)])]));
    }

    #[test]
    fn equality_is_order_sensitive_only_against_another_ordereddict() {
        let ab = build(pairs(&[("a", 1), ("b", 2)]));
        let ba = build(pairs(&[("b", 2), ("a", 1)]));
        // OrderedDict([("a",1),("b",2)]) == OrderedDict([("b",2),("a",1)]) -> False
        assert!(ab != ba);
        // OrderedDict([("a",1),("b",2)]) == OrderedDict([("a",1),("b",2)]) -> True
        assert!(ab == build(pairs(&[("a", 1), ("b", 2)])));
        // OrderedDict([("a",1),("b",2)]) == {"b": 2, "a": 1} -> True, and the reverse
        let d = dict(&[("b", 2), ("a", 1)]);
        let plain = PyValue::Dict(std::sync::Arc::new(d.clone()));
        assert!(ab == plain);
        assert!(plain == ab);
        assert!(ab == d);
        // OrderedDict([("a",1)]) == {"a": 2} -> False
        let other = PyValue::Dict(std::sync::Arc::new(dict(&[("a", 2)])));
        assert!(build(pairs(&[("a", 1)])) != other);
        // [("a", 1)] == OrderedDict([("a", 1)]) -> False; OrderedDict() == () -> False;
        // OrderedDict() == None -> False
        assert!(pairs(&[("a", 1)]) != build(pairs(&[("a", 1)])));
        assert!(build(tup(vec![])) != tup(vec![]));
        assert!(build(tup(vec![])) != PyValue::None_);
    }

    #[test]
    fn truthiness_len_iteration_membership_and_index() {
        let od = build(pairs(&[("b", 1), ("a", 2), ("c", 3)]));
        // bool(OrderedDict()) -> False; bool(OrderedDict([("b", 1)])) -> True
        assert!(!build(tup(vec![])).is_truthy());
        assert!(od.is_truthy());
        // len(OrderedDict([("b",1),("a",2),("c",3)])) -> 3
        assert_eq!(od.len(), 3);
        // list(OrderedDict([("b",1),("a",2),("c",3)])) -> ['b', 'a', 'c']
        let keys: Vec<PyValue> = od.clone().into_iter().collect();
        assert_eq!(keys, vec![s("b"), s("a"), s("c")]);
        // "a" in od -> True; "q" in od -> False; 1 in od -> False
        assert!(od.py_contains(&s("a")));
        assert!(!od.py_contains(&s("q")));
        assert!(!od.py_contains(&PyValue::Int(1)));
        // od["a"] -> 2
        assert_eq!(od.py_index("a").unwrap(), PyValue::Int(2));
        // od["nope"] -> KeyError: 'nope' ; od[1] -> KeyError: 1
        assert_eq!(od.py_index("nope").unwrap_err().message, "'nope'");
        assert_eq!(od.py_index(1i64).unwrap_err().message, "1");
        // od["b"] = 10 through a boxed index-mut keeps b's position
        let mut od = od;
        *od.py_index_mut("b").unwrap() = PyValue::Int(10);
        assert_eq!(od.py_repr(), "OrderedDict({'b': 10, 'a': 2, 'c': 3})");
    }

    #[test]
    fn type_name_and_unhashability() {
        // type(OrderedDict()).__name__ -> 'OrderedDict'
        assert_eq!(build(tup(vec![])).py_type_name(), "OrderedDict");
        // hash(OrderedDict()) -> TypeError: unhashable type: 'collections.OrderedDict'
        let od = build(tup(vec![]));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use std::hash::Hash;
            let mut h = std::collections::hash_map::DefaultHasher::new();
            od.hash(&mut h);
        }));
        let payload = caught.unwrap_err();
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|m| m.to_string()))
            .unwrap();
        assert_eq!(msg, "TypeError: unhashable type: 'collections.OrderedDict'");
    }

    #[test]
    fn typed_ordereddict_boxes_without_losing_its_type() {
        // A typed OrderedDict flowing into a boxed slot stays an OrderedDict.
        let typed = OrderedDict::from_pairs(vec![("b".to_string(), 1i64), ("a".to_string(), 2)]);
        let boxed = PyValue::from(typed);
        assert_eq!(boxed.py_repr(), "OrderedDict({'b': 1, 'a': 2})");
        // A list of pairs boxes as a tuple of 2-tuples (the list-as-tuple form).
        assert_eq!(PyValue::from(vec![("a", 1i64)]), pairs(&[("a", 1)]));
    }

    #[test]
    fn json_and_urlencode_treat_it_as_a_mapping() {
        // json.dumps(OrderedDict([("b", 1), ("a", [1, 2])])) -> {"b": 1, "a": [1, 2]}
        let od = build(tup(vec![
            pair("b", 1),
            tup(vec![s("a"), tup(vec![PyValue::Int(1), PyValue::Int(2)])]),
        ]));
        assert_eq!(
            stdpython::json::dumps_pyvalue(od, None).unwrap(),
            "{\"b\": 1, \"a\": [1, 2]}"
        );
        // urllib.parse.urlencode(OrderedDict([("b", 1), ("a", 2)])) -> b=1&a=2
        let q = build(pairs(&[("b", 1), ("a", 2)]));
        assert_eq!(stdpython::urllib::parse::urlencode(&q, false).unwrap(), "b=1&a=2");
    }
}


// ── Container operations on a boxed value (issue #137 frontier) ─────────────
//
// A value inference files into a `PyValue` slot lost its static type, so
// `d.get(k)`, `d.items()`, `d.pop(k)` and `xs[a:b]` on it had no method to
// call and the generated crate failed to compile (111 such sites in the #137
// corpus at 28efebd). Each test carries the CPython expression it pins,
// verified against python3 3.14.1.

mod boxed_container_ops {
    use stdpython::{
        PyContains, PyDict, PyDictOps, PyPop, PySetIndex, PySlice, PySliceReplace, PyValue,
    };

    fn boxed_dict(pairs: &[(&str, &str)]) -> PyValue {
        let mut d = PyDict::new();
        for (k, v) in pairs {
            PySetIndex::py_set_index(&mut d, k.to_string(), PyValue::Str(v.to_string())).unwrap();
        }
        PyValue::Dict(std::sync::Arc::new(d))
    }

    #[test]
    fn get_hit_miss_and_default() {
        // d = {"a": "1", "b": "2"}; d.get("a") == "1"; d.get("zz") is None;
        // d.get("zz", "D") == "D"
        let d = boxed_dict(&[("a", "1"), ("b", "2")]);
        assert_eq!(PyDictOps::py_get(&d, &"a".to_string()), Some(PyValue::Str("1".into())));
        assert_eq!(PyDictOps::py_get(&d, &"zz".to_string()), None);
        assert_eq!(
            PyDictOps::py_get_default(&d, &"zz".to_string(), PyValue::Str("D".into())),
            PyValue::Str("D".into())
        );
    }

    #[test]
    fn items_keys_and_values_keep_insertion_order() {
        // dict({"b": "2", "a": "1"}).items() == [("b","2"), ("a","1")]
        let d = boxed_dict(&[("b", "2"), ("a", "1")]);
        assert_eq!(PyDictOps::py_items(&d), vec![
            ("b".to_string(), PyValue::Str("2".into())),
            ("a".to_string(), PyValue::Str("1".into())),
        ]);
        assert_eq!(PyDictOps::py_keys(&d), vec!["b".to_string(), "a".to_string()]);
        assert_eq!(
            PyDictOps::py_values(&d),
            vec![PyValue::Str("2".into()), PyValue::Str("1".into())]
        );
    }

    #[test]
    fn pop_removes_and_raises_cpython_keyerror_on_a_miss() {
        // d = {"a": "1", "b": "2"}; d.pop("a") == "1", leaving {"b": "2"};
        // d.pop("zz") raises KeyError: 'zz'
        let mut d = boxed_dict(&[("a", "1"), ("b", "2")]);
        assert_eq!(PyPop::py_pop(&mut d, "a".to_string()).unwrap(), PyValue::Str("1".into()));
        assert_eq!(PyDictOps::py_items(&d), vec![("b".to_string(), PyValue::Str("2".into()))]);
        let err = PyPop::py_pop(&mut d, "zz".to_string()).unwrap_err();
        assert_eq!(err.exception_type, "KeyError");
        assert!(
            err.message.contains("zz"),
            "message was {:?}",
            err.message
        );
    }

    #[test]
    fn setdefault_inserts_only_when_absent() {
        // d = {"a": "1"}; d.setdefault("n", "N") == "N" and stores it;
        // d.setdefault("a", "OTHER") == "1" (the stored value wins)
        let mut d = boxed_dict(&[("a", "1")]);
        assert_eq!(
            PyDictOps::py_setdefault(&mut d, "n".to_string(), PyValue::Str("N".into())),
            PyValue::Str("N".into())
        );
        assert_eq!(PyDictOps::py_get(&d, &"n".to_string()), Some(PyValue::Str("N".into())));
        assert_eq!(
            PyDictOps::py_setdefault(&mut d, "a".to_string(), PyValue::Str("OTHER".into())),
            PyValue::Str("1".into())
        );
    }

    #[test]
    fn set_index_then_contains() {
        // d = {}; d["k"] = "v"; "k" in d is True; "z" in d is False
        let mut d = boxed_dict(&[]);
        PySetIndex::py_set_index(&mut d, "k".to_string(), PyValue::Str("v".into())).unwrap();
        assert_eq!(PyDictOps::py_get(&d, &"k".to_string()), Some(PyValue::Str("v".into())));
        assert!(PyContains::py_contains(&d, "k"));
        assert!(!PyContains::py_contains(&d, "z"));
    }

    #[test]
    fn str_slices_by_character_and_bytes_by_octet() {
        // "abcdef"[1:3] == "bc"; [::2] == "ace"; [::-1] == "fedcba";
        // [9:99] == "" (bounds clamp, never raise); [-2:] == "ef"
        let s = PyValue::Str("abcdef".into());
        assert_eq!(s.py_slice(Some(1), Some(3), None), PyValue::Str("bc".into()));
        assert_eq!(s.py_slice(None, None, Some(2)), PyValue::Str("ace".into()));
        assert_eq!(s.py_slice(None, None, Some(-1)), PyValue::Str("fedcba".into()));
        assert_eq!(s.py_slice(Some(9), Some(99), None), PyValue::Str("".into()));
        assert_eq!(s.py_slice(Some(-2), None, None), PyValue::Str("ef".into()));
        // b"abcdef"[1:3] == b"bc" — bytes slice by OCTET, the line len draws.
        let b = PyValue::Bytes(b"abcdef".to_vec());
        assert_eq!(b.py_slice(Some(1), Some(3), None), PyValue::Bytes(b"bc".to_vec()));
    }

    #[test]
    fn a_tuple_member_stays_a_tuple_when_sliced() {
        // (1, 2, 3, 4)[1:3] == (2, 3) — a tuple, never a list.
        let t = PyValue::Tuple(std::sync::Arc::new(vec![
            PyValue::Int(1),
            PyValue::Int(2),
            PyValue::Int(3),
            PyValue::Int(4),
        ]));
        let sliced = t.py_slice(Some(1), Some(3), None);
        assert_eq!(
            sliced,
            PyValue::Tuple(std::sync::Arc::new(vec![PyValue::Int(2), PyValue::Int(3)]))
        );
        assert_eq!(stdpython::py_value_repr(&sliced), "(2, 3)");
    }

    #[test]
    fn a_non_container_member_refuses_rather_than_returning_a_wrong_value() {
        // 5.get("a", 9) and "x".items() are AttributeError in CPython. The
        // dict trait is INFALLIBLE, so these panic with the refusal instead of
        // returning the default or an empty list — a default here would be
        // indistinguishable from a real miss and would diverge silently.
        let n = PyValue::Int(5);
        let k = "a".to_string();
        let d = 9i64;
        // AssertUnwindSafe: the trait carries interior mutability, but these
        // closures only borrow an immutable receiver and own their arguments.
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let key = k.clone();
                PyDictOps::py_get_default(&PyValue::Int(5), &key, PyValue::Int(d))
            }))
            .is_err()
        );
        assert!(std::panic::catch_unwind(|| PyDictOps::py_get(&PyValue::Int(5), &k)).is_err());
        assert!(std::panic::catch_unwind(|| PyDictOps::py_items(&PyValue::Int(5))).is_err());
        assert!(std::panic::catch_unwind(|| PyDictOps::py_keys(&PyValue::Int(5))).is_err());
        assert!(std::panic::catch_unwind(|| PyDictOps::py_values(&PyValue::Int(5))).is_err());
        // setdefault and update must refuse too, never silently no-op.
        assert!(std::panic::catch_unwind(|| {
            let mut s = PyValue::Str("x".into());
            let key = k.clone();
            PyDictOps::py_setdefault(&mut s, key, PyValue::Int(1))
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            let mut n2 = PyValue::Int(5);
            PyDictOps::update(&mut n2, PyDict::new())
        })
        .is_err());
        // The fallible entry point carries CPython's TypeError by name.
        let err = n.not_a_container("items()");
        assert_eq!(err.exception_type, "TypeError");
        assert!(err.message.contains("int"), "message was {:?}", err.message);
    }

    #[test]
    fn a_tuple_member_is_immutable_under_slice_assignment() {
        // t = (1, 2); t[0:1] = [9] raises TypeError: 'tuple' object does not
        // support item assignment — a boxed tuple must NOT be edited in place,
        // and del t[0:1] raises TypeError the same way.
        let mut t = PyValue::Tuple(std::sync::Arc::new(vec![
            PyValue::Int(1),
            PyValue::Int(2),
        ]));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                PySliceReplace::py_slice_assign(&mut t, Some(0), Some(1), vec![PyValue::Int(9)]);
            }))
            .is_err(),
            "a boxed tuple must refuse slice assignment"
        );
        // The stepped (fallible) arm reports the same refusal as an exception.
        let mut t2 = PyValue::Tuple(std::sync::Arc::new(vec![PyValue::Int(1)]));
        let err = PySliceReplace::py_slice_assign_step(
            &mut t2,
            None,
            None,
            2,
            vec![PyValue::Int(9)],
        )
        .unwrap_err();
        assert_eq!(err.exception_type, "TypeError");
        // Deletion refuses too.
        let mut t3 = PyValue::Tuple(std::sync::Arc::new(vec![PyValue::Int(1)]));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PySliceReplace::py_slice_delete(&mut t3, None, None);
        }))
        .is_err());
    }

    #[test]
    fn a_range_member_slices_to_a_range() {
        // range(6)[1:4] == range(1, 4); [::2] == range(0, 6, 2);
        // [::-1] == range(5, -1, -1); [9:99] == range(6, 6)
        let r = PyValue::from(stdpython::range(6));
        assert_eq!(stdpython::py_value_repr(&r.py_slice(Some(1), Some(4), None)), "range(1, 4)");
        assert_eq!(
            stdpython::py_value_repr(&r.py_slice(None, None, Some(2))),
            "range(0, 6, 2)"
        );
        assert_eq!(
            stdpython::py_value_repr(&r.py_slice(None, None, Some(-1))),
            "range(5, -1, -1)"
        );
        assert_eq!(
            stdpython::py_value_repr(&r.py_slice(Some(9), Some(99), None)),
            "range(6, 6)"
        );
        // A step-2 range keeps its own step: range(0, 10, 2)[::2] == range(0, 10, 4)
        let r2 = PyValue::from(stdpython::range_start_stop_step(0, 10, 2).unwrap());
        assert_eq!(
            stdpython::py_value_repr(&r2.py_slice(None, None, Some(2))),
            "range(0, 10, 4)"
        );
    }

    #[test]
    fn an_empty_range_slice_keeps_cpython_s_start_and_step() {
        // range(6)[2:4:-1] == range(2, 4, -1) — CPython keeps the NORMALISED
        // start and step even when nothing is selected, so the repr differs
        // from the range(4, 4) a naive empty branch would build.
        let r = PyValue::from(stdpython::range(6));
        assert_eq!(
            stdpython::py_value_repr(&r.py_slice(Some(2), Some(4), Some(-1))),
            "range(2, 4, -1)"
        );
        // range(6)[4:2] == range(4, 2): empty forward, start and stop kept.
        assert_eq!(
            stdpython::py_value_repr(&r.py_slice(Some(4), Some(2), None)),
            "range(4, 2)"
        );
        // range(6)[4:2:-1] == range(4, 2, -1): also empty, step preserved.
        assert_eq!(
            stdpython::py_value_repr(&r.py_slice(Some(4), Some(2), Some(-1))),
            "range(4, 2, -1)"
        );
    }

    #[test]
    fn a_range_slice_past_the_i64_endpoint_refuses_rather_than_wrapping() {
        // CPython's range(2**63-2, 2**63-1, 2)[:] is
        // range(9223372036854775806, 9223372036854775808, 2). Its repr needs an
        // endpoint of 2**63 — outside PyRange's i64 fields.
        //
        // The old code cast `py_len()` (a usize) to i64 up front, so a length
        // above i64::MAX wrapped NEGATIVE and a full slice silently became the
        // EMPTY range. The arithmetic is now done in i128, and `narrow` refuses
        // an endpoint that cannot be represented rather than wrapping to a
        // range different from CPython's.
        let big = stdpython::range_start_stop_step(i64::MAX - 2, i64::MAX - 1, 2).unwrap();
        assert_eq!(big.py_len(), 1);
        // The full slice's endpoints are i128-only: next + len*step is 2**63-1,
        // which fits, but CPython's exclusive stop is one element PAST it. The
        // arithmetic must therefore stay wide; if it were narrowed to i64 the
        // result would be a different (wrong) range, so this asserts the
        // element is still there rather than that it panicked.
        let full = big.py_slice(None, None, None);
        assert_eq!(full.py_len(), 1, "a full slice must keep the element");
        assert_eq!(
            full.clone().next(),
            Some(i64::MAX - 2),
            "the element must be the range's first value, not a wrapped one"
        );
        assert!(full.py_contains(&(i64::MAX - 2)));
        // A boundary range whose whole slice fits works cleanly: CPython's
        // range(2**63-4, 2**63-2, 2) holds one value and slices to it.
        let edge = stdpython::range_start_stop_step(i64::MAX - 3, i64::MAX - 1, 2).unwrap();
        assert_eq!(edge.py_slice(None, None, None).py_len(), 1);
    }

    #[test]
    fn a_zero_slice_step_is_cpythons_value_error() {
        // xs[::0] raises ValueError: slice step cannot be zero — on READS too.
        let r = PyValue::from(stdpython::range(6));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                r.py_slice(None, None, Some(0))
            }))
            .is_err(),
            "a zero step must raise, not divide by zero"
        );
        // On the STEP mutation arms it is a typed ValueError. (For an
        // immutable receiver CPython reports the TypeError first — verified
        // with `t = (1,); t[::0] = []` — so this pins the step check itself.)
        let mut t = PyValue::Tuple(std::sync::Arc::new(vec![PyValue::Int(1)]));
        let err = PySliceReplace::py_slice_assign_step(&mut t, None, None, 0, vec![]).unwrap_err();
        assert_eq!(err.exception_type, "ValueError");
        assert!(err.message.contains("step"), "message was {:?}", err.message);
        let err = PySliceReplace::py_slice_delete_step(&mut t, None, None, 0).unwrap_err();
        assert_eq!(err.exception_type, "ValueError");
    }
}
/// Issue #434: the numeric tower (`1 == 1.0 == True == 1+0j`) holds INSIDE
/// boxed containers, not only at the top-level `py_value_eq`. Every
/// expected value below was captured from python3.12.
mod boxed_numeric_equality {
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;
    use stdpython::collections::OrderedDict;
    use stdpython::*;

    fn s(v: &str) -> PyValue {
        PyValue::Str(v.to_string())
    }
    fn tup(items: Vec<PyValue>) -> PyValue {
        PyValue::Tuple(Arc::new(items))
    }
    fn dict(items: Vec<(&str, PyValue)>) -> PyValue {
        PyValue::Dict(Arc::new(
            items.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        ))
    }
    fn od(items: Vec<(&str, PyValue)>) -> PyValue {
        OrderedDict::from_boxed(dict(items)).unwrap()
    }
    fn h(v: &PyValue) -> u64 {
        let mut st = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut st);
        st.finish()
    }
    fn cplx(re: f64, im: f64) -> PyValue {
        PyValue::Complex(Complex::new(re, im))
    }

    #[test]
    fn scalars() {
        // True == 1 -> True; 1 == 1+0j -> True; 1.0 == 1+0j -> True;
        // True == 1+0j -> True; 1 == 1+1j -> False; 0.5 == 1 -> False
        assert!(PyValue::Bool(true) == PyValue::Int(1));
        assert!(PyValue::Int(1) == PyValue::Bool(true));
        assert!(PyValue::Int(1) == cplx(1.0, 0.0));
        assert!(cplx(1.0, 0.0) == PyValue::Float(1.0));
        assert!(PyValue::Bool(true) == cplx(1.0, 0.0));
        assert!(PyValue::Int(1) != cplx(1.0, 1.0));
        assert!(PyValue::Float(0.5) != PyValue::Int(1));
        // Float(nan) != Float(nan) stays.
        assert!(PyValue::Float(f64::NAN) != PyValue::Float(f64::NAN));
    }

    #[test]
    fn inside_tuple() {
        // (1, "x") == (1.0, "x") -> True; (1,) == (1.5,) -> False
        assert!(tup(vec![PyValue::Int(1), s("x")]) == tup(vec![PyValue::Float(1.0), s("x")]));
        assert!(tup(vec![PyValue::Int(1)]) != tup(vec![PyValue::Float(1.5)]));
        // (1, "x") != (1.0, "x") -> False
        assert!(!(tup(vec![PyValue::Int(1), s("x")]) != tup(vec![PyValue::Float(1.0), s("x")])));
    }

    #[test]
    fn inside_dict_and_ordereddict() {
        // {"n": 1, "s": "x"} == {"n": 1.0, "s": "x"} -> True (the issue)
        assert!(
            dict(vec![("n", PyValue::Int(1)), ("s", s("x"))])
                == dict(vec![("n", PyValue::Float(1.0)), ("s", s("x"))])
        );
        // {"n": True} == {"n": 1} -> True
        assert!(dict(vec![("n", PyValue::Bool(true))]) == dict(vec![("n", PyValue::Int(1))]));
        // OrderedDict(n=1) == OrderedDict(n=1.0) -> True
        assert!(od(vec![("n", PyValue::Int(1))]) == od(vec![("n", PyValue::Float(1.0))]));
        // OrderedDict(n=1) == {"n": 1.0} -> True (both directions)
        assert!(od(vec![("n", PyValue::Int(1))]) == dict(vec![("n", PyValue::Float(1.0))]));
        assert!(dict(vec![("n", PyValue::Float(1.0))]) == od(vec![("n", PyValue::Int(1))]));
    }

    #[test]
    fn nested() {
        // (1, (2, "a")) == (1.0, (2.0, "a")) -> True
        assert!(
            tup(vec![PyValue::Int(1), tup(vec![PyValue::Int(2), s("a")])])
                == tup(vec![PyValue::Float(1.0), tup(vec![PyValue::Float(2.0), s("a")])])
        );
        // {"k": (1, 2)} == {"k": (1.0, 2.0)} -> True
        assert!(
            dict(vec![("k", tup(vec![PyValue::Int(1), PyValue::Int(2)]))])
                == dict(vec![("k", tup(vec![PyValue::Float(1.0), PyValue::Float(2.0)]))])
        );
        // {"k": {"j": 1}} == {"k": {"j": True}} -> True
        assert!(
            dict(vec![("k", dict(vec![("j", PyValue::Int(1))]))])
                == dict(vec![("k", dict(vec![("j", PyValue::Bool(true))]))])
        );
    }

    #[test]
    fn hash_agrees_with_equality() {
        // hash(1) == hash(1.0) == hash(True) == hash(1+0j) -> True
        let one = [
            PyValue::Int(1),
            PyValue::Float(1.0),
            PyValue::Bool(true),
            cplx(1.0, 0.0),
        ];
        for v in &one {
            assert_eq!(h(v), h(&one[0]), "{v:?}");
        }
        // hash(0) == hash(False) == hash(0.0) == hash(-0.0) -> True
        let zero = [
            PyValue::Int(0),
            PyValue::Bool(false),
            PyValue::Float(0.0),
            PyValue::Float(-0.0),
            cplx(0.0, 0.0),
        ];
        for v in &zero {
            assert_eq!(h(v), h(&zero[0]), "{v:?}");
        }
        // hash(2.5) == hash(2) -> False
        assert_ne!(h(&PyValue::Float(2.5)), h(&PyValue::Int(2)));
        // Hash recurses through a tuple key.
        assert_eq!(
            h(&tup(vec![PyValue::Int(1), s("x")])),
            h(&tup(vec![PyValue::Float(1.0), s("x")]))
        );
    }

    #[test]
    fn dict_keys_and_sets() {
        // {1: 'a'}[1.0] -> 'a'; {1: 'a', 1.0: 'b'} -> {1: 'b'};
        // {1: 'a', True: 'b'} -> {1: 'b'}; {1, 1.0, True} -> {1}
        let mut d: PyDict<PyValue, &str> = PyDict::default();
        d.insert(PyValue::Int(1), "a");
        assert_eq!(d.get(&PyValue::Float(1.0)), Some(&"a"));
        d.insert(PyValue::Float(1.0), "b");
        d.insert(PyValue::Bool(true), "c");
        assert_eq!(d.len(), 1);
        assert_eq!(d.get(&PyValue::Int(1)), Some(&"c"));
        let set: std::collections::HashSet<PyValue> =
            [PyValue::Int(1), PyValue::Float(1.0), PyValue::Bool(true)]
                .into_iter()
                .collect();
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn int_float_comparison_is_exact() {
        // 2**53 + 1 == float(2**53) -> False; 2**53 == float(2**53) -> True;
        // (2**53 + 1,) == (float(2**53),) -> False
        let big = 9007199254740992i64;
        assert!(PyValue::Int(big + 1) != PyValue::Float(big as f64));
        assert!(PyValue::Int(big) == PyValue::Float(big as f64));
        assert!(tup(vec![PyValue::Int(big + 1)]) != tup(vec![PyValue::Float(big as f64)]));
        // i64::MAX == 9.223372036854775807e18 -> False;
        // i64::MIN == -9.223372036854775808e18 -> True
        assert!(PyValue::Int(i64::MAX) != PyValue::Float(9.223372036854775807e18));
        assert!(PyValue::Int(i64::MIN) == PyValue::Float(-9.223372036854775808e18));
    }

    #[test]
    fn equal_ranges_hash_alike() {
        // (range(0, 3, 2),) == (range(0, 4, 2),) -> True;
        // hash(range(0, 3, 2)) == hash(range(0, 4, 2)) -> True
        let a = PyValue::Range(range_start_stop_step(0, 3, 2).unwrap());
        let b = PyValue::Range(range_start_stop_step(0, 4, 2).unwrap());
        assert!(tup(vec![a.clone()]) == tup(vec![b.clone()]));
        assert_eq!(h(&a), h(&b));
    }

    #[test]
    fn list_neighbours() {
        // l = [1, "x", 2.0, True]
        // 1.0 in l -> True; l.index(2) -> 2; l.count(1) -> 2; l.count(1.0) -> 2
        // l.index(True) -> 0; 2 in l -> True; (2.0,) in [(2,)] -> True
        let l = vec![PyValue::Int(1), s("x"), PyValue::Float(2.0), PyValue::Bool(true)];
        assert!(l.py_contains(&PyValue::Float(1.0)));
        assert_eq!(l.py_index_of(&PyValue::Int(2)).unwrap(), 2);
        assert_eq!(l.count(&PyValue::Int(1)), 2);
        assert_eq!(l.count(&PyValue::Float(1.0)), 2);
        assert_eq!(l.py_index_of(&PyValue::Bool(true)).unwrap(), 0);
        assert!(l.py_contains(&PyValue::Int(2)));
        assert!(vec![tup(vec![PyValue::Int(2)])].py_contains(&tup(vec![PyValue::Float(2.0)])));
        // [1, "x"] == [1.0, "x"] -> True; != -> False
        let a = vec![PyValue::Int(1), s("x")];
        let b = vec![PyValue::Float(1.0), s("x")];
        assert!(a == b);
        assert!(!(a != b));
    }
}

mod chainmap_ordering {
    use stdpython::collections::ChainMap;
    use stdpython::{Len, PyDict};

    fn pd(pairs: &[(&'static str, i64)]) -> PyDict<&'static str, i64> {
        PyDict::from_iter(pairs.iter().copied())
    }

    #[test]
    fn iteration_walks_the_maps_in_reverse_first_seen_wins() {
        // Verified against python3 (identical under PYTHONHASHSEED=0, 1, 999):
        // cm = ChainMap({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
        let cm = ChainMap::new(vec![pd(&[("a", 1), ("b", 2)]), pd(&[("b", 3), ("c", 4)])]);
        // list(cm.keys())   == ['b', 'c', 'a']  — the LAST map's keys first
        assert_eq!(cm.keys(), vec!["b", "c", "a"]);
        // list(cm.values()) == [2, 4, 1]        — first map wins for 'b'
        assert_eq!(cm.values(), vec![2, 4, 1]);
        // list(cm.items())  == [('b', 2), ('c', 4), ('a', 1)]
        assert_eq!(cm.items(), vec![("b", 2), ("c", 4), ("a", 1)]);
        // len(cm) == 3
        assert_eq!(cm.len(), 3);

        // cm3 = ChainMap({'x': 1}, {'y': 2, 'x': 9}, {'z': 3, 'y': 8})
        let cm3 = ChainMap::new(vec![
            pd(&[("x", 1)]),
            pd(&[("y", 2), ("x", 9)]),
            pd(&[("z", 3), ("y", 8)]),
        ]);
        // list(cm3) == ['z', 'y', 'x'];  list(cm3.values()) == [3, 2, 1]
        assert_eq!(cm3.keys(), vec!["z", "y", "x"]);
        assert_eq!(cm3.values(), vec![3, 2, 1]);
    }

    #[test]
    fn order_is_deterministic_not_hash_randomized() {
        // The HashMap/HashSet version produced a different key order on
        // different runs of the same program. Rebuilding the same
        // ChainMap must always yield CPython's one order.
        for _ in 0..50 {
            let cm = ChainMap::new(vec![pd(&[("a", 1), ("b", 2)]), pd(&[("b", 3), ("c", 4)])]);
            assert_eq!(cm.keys(), vec!["b", "c", "a"]);
        }
    }

    #[test]
    fn deleting_from_the_first_map_keeps_the_rest_in_order() {
        // d = ChainMap({'p': 1, 'q': 2, 'r': 3}, {'s': 4}); del d['p']
        // list(d) == ['s', 'q', 'r']
        // A swap_remove would move 'r' into 'p''s slot, giving
        // ['s', 'r', 'q'] — the first map is a dict, which keeps order.
        let mut d = ChainMap::new(vec![pd(&[("p", 1), ("q", 2), ("r", 3)]), pd(&[("s", 4)])]);
        assert_eq!(d.remove(&"p"), Some(1));
        assert_eq!(d.keys(), vec!["s", "q", "r"]);
    }

    #[test]
    fn writes_go_to_the_first_map_and_shadow_later_ones() {
        // w = ChainMap({'k': 1}, {'k': 2, 'm': 3}); w['k'] = 10
        // dict(w) == {'k': 10, 'm': 3}
        let mut w = ChainMap::new(vec![pd(&[("k", 1)]), pd(&[("k", 2), ("m", 3)])]);
        w.insert("k", 10);
        assert_eq!(w.items(), vec![("k", 10), ("m", 3)]);
    }

    #[test]
    fn new_child_is_pushed_on_the_front_and_iterates_last() {
        // c = ChainMap({'a': 1}).new_child({'b': 2})
        // list(c) == ['a', 'b'];  c.maps == [{'b': 2}, {'a': 1}]
        let mut c = ChainMap::new(vec![pd(&[("a", 1)])]);
        c.new_child(pd(&[("b", 2)]));
        assert_eq!(c.keys(), vec!["a", "b"]);
        assert_eq!(c.num_maps(), 2);
    }
}

#[test]
fn numpy_array_scalar_comparisons_broadcast_and_keep_operand_order() {
    use stdpython::stdlib::numpy;
    let a = numpy::array(vec![1.0, 5.0, 2.0, 8.0]);
    // np.greater(a, 2.0) -> [False  True False  True]
    let m = numpy::greater(&a, &2.0).unwrap();
    assert_eq!(m.to_string(), "[False  True False  True]");
    // a[np.greater(a, 2.0)] -> [5. 8.]
    assert_eq!(a.py_index(m).unwrap().to_string(), "[5. 8.]");
    // np.greater(2.0, a) -> [ True False False False]
    let r = numpy::greater(&2.0, &a).unwrap();
    assert_eq!(r.to_string(), "[ True False False False]");
    // np.greater(np.array([[1, 2], [3, 4]]), 2) -> [[False False]\n [ True  True]]
    let g = numpy::greater(&numpy::array(vec![vec![1i64, 2], vec![3, 4]]), &2i64).unwrap();
    assert_eq!(g.to_string(), "[[False False]\n [ True  True]]");
    // np.equal(np.array([0, 1, 2]), True) -> [False  True False]
    let e = numpy::equal(&numpy::array(vec![0i64, 1, 2]), &true).unwrap();
    assert_eq!(e.to_string(), "[False  True False]");
}

// Counter construction, added for issue #137's charset_normalizer frontier
// (`Counter(layer)` in coherence detection, which the round-100 sweep could
// not convert at all). Every expectation below was produced by running the
// quoted Python under python3 with PYTHONHASHSEED=0.
mod counter_construction {
    use super::*;
    use stdpython::collections::Counter;

    fn pd(pairs: &[(&'static str, i64)]) -> PyDict<&'static str, i64> {
        PyDict::from_iter(pairs.iter().copied())
    }

    #[test]
    fn counting_an_iterable_is_distinct_from_a_mappings_values() {
        // c = Counter("aab"); c.get('a'), c.get('b')
        let c = Counter::from_iter("aab".chars());
        assert_eq!(c.get(&'a'), 2);
        assert_eq!(c.get(&'b'), 1);
        // d = Counter({'a': 3, 'b': 1}); d.get('a')
        //
        // CPython dispatches __init__ on has_key: a MAPPING's VALUES are the
        // counts, so this is 3, not 1 (counting the keys would give 1).
        let d: Counter<&'static str> = Counter::from_mapping(&pd(&[("a", 3), ("b", 1)]));
        assert_eq!(d.get(&"a"), 3);
        // f = Counter(['x', 'y', 'x']); f.get('x')
        let f: Counter<char> = Counter::from_iter(['x', 'y', 'x'].into_iter());
        assert_eq!(f.get(&'x'), 2);
        // e = Counter(); e.get('x', 5) -> 5 (an empty Counter has no keys)
        let e: Counter<char> = Counter::new();
        assert_eq!(e.get_opt(&'x'), None);
        assert_eq!(e.get_opt(&'x').unwrap_or(5), 5);
    }

    #[test]
    fn get_is_dict_get_so_a_missing_key_is_none_not_zero() {
        // The rule this pins is the one a Counter inherits from dict, and it
        // is NOT the __missing__ rule: only `c[k]` answers 0.
        // c = Counter("aab"); c.get('z') -> None      but c['z'] -> 0
        let c = Counter::from_iter("aab".chars());
        assert_eq!(c.get_opt(&'z'), None);
        assert_eq!(c.get(&'z'), 0);
        // c.get('z', 99) -> 99; c.get('z', None) -> None
        assert_eq!(c.get_opt(&'z').unwrap_or(99), 99);
        assert_eq!(c.get_opt(&'z').unwrap_or(-1), -1);
        // c.get('a') -> 2
        assert_eq!(c.get_opt(&'a'), Some(2));
    }

    #[test]
    fn subscripting_is_missing_so_a_missing_key_counts_zero() {
        // The other half of the pair above, and the reason the two disagree:
        // `__missing__` answers 0 where `dict.get` answers None.
        // c = Counter('aab'); c['a'] -> 2, c['z'] -> 0
        let c: Counter<char> = Counter::from_iter("aab".chars());
        assert_eq!(crate::PyIndex::py_index(&c, 'a').unwrap(), 2);
        assert_eq!(crate::PyIndex::py_index(&c, 'z').unwrap(), 0);
        // The String-keyed arm takes a literal, which lowers as `&str`.
        // s = Counter('aab'); s['a'] -> 2, s['z'] -> 0
        let s: Counter<String> = Counter::from_iter("aab".chars().map(String::from));
        assert_eq!(crate::PyIndex::py_index(&s, "a").unwrap(), 2);
        assert_eq!(crate::PyIndex::py_index(&s, "z").unwrap(), 0);
    }

    #[test]
    fn most_common_takes_an_optional_n_where_absent_means_all() {
        // c = Counter("aab"); c.most_common() -> [('a', 2), ('b', 1)]
        let c = Counter::from_iter("aab".chars());
        assert_eq!(format!("{:?}", c.most_common(None)), "[('a', 2), ('b', 1)]");
        // c.most_common(1) -> [('a', 2)]
        assert_eq!(format!("{:?}", c.most_common(Some(1))), "[('a', 2)]");
        // CPython sorts by COUNT and breaks ties by first insertion, so a
        // stable sort on the count alone reproduces the order.
        // g = Counter("bbaa"); g.most_common() -> [('b', 2), ('a', 2)]
        let g = Counter::from_iter("bbaa".chars());
        assert_eq!(format!("{:?}", g.most_common(None)), "[('b', 2), ('a', 2)]");
    }
}

#[test]
fn set_remove_raises_key_error_with_the_element_repr() {
    // s = {"cat", "dog"}; s.remove("cat"); s.remove("cat")
    //   -> KeyError: 'cat'        (str(e) == "'cat'")
    // n = {3}; n.remove(4)        -> KeyError: 4
    let mut s = std::collections::HashSet::from(["cat".to_string(), "dog".to_string()]);
    py_set_remove(&mut s, &"cat".to_string()).unwrap();
    assert_eq!(len(&s), 1);
    let e = py_set_remove(&mut s, &"cat".to_string()).unwrap_err();
    assert_eq!(e.exception_type, "KeyError");
    assert_eq!(e.message, "'cat'");
    let mut n = std::collections::HashSet::from([3i64]);
    assert_eq!(py_set_remove(&mut n, &4).unwrap_err().message, "4");
    // len() counts members of any set: len({3}) == 1
    assert_eq!(len(&n), 1);
}
