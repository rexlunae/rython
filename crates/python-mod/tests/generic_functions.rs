//! python_module! integration test: call GENERIC inferred functions from
//! Rust (issue #109, M5). `src/math_ops.py` and `src/string_ops.py` have
//! UNANNOTATED parameters, so the macro's conversion emits trait-bound
//! generic signatures — `fn add<A, B>(a: A, b: B) -> Result<A, PyException>
//! where A: PyAdd<B>`. Calling them from Rust must monomorphize per call
//! site exactly like Python's dynamic dispatch, with no turbofish.
use python_mod::python_module;

python_module!(math_ops);
python_module!(string_ops);

#[test]
fn generic_add_monomorphizes_from_rust() {
    // int + int, float + float, str + str — one generic function, three
    // instantiations (Python's `add(1, 2)` / `add(1.5, 2.5)` /
    // `add("ab", "cd")`).
    assert_eq!(math_ops::add(1i64, 2i64).unwrap(), 3i64);
    assert_eq!(math_ops::add(1.5f64, 2.5f64).unwrap(), 4.0f64);
    assert_eq!(
        math_ops::add("ab".to_string(), "cd".to_string()).unwrap(),
        "abcd"
    );
}

#[test]
fn generic_multiply_and_recursion_from_rust() {
    assert_eq!(math_ops::multiply(4i64, 6i64).unwrap(), 24i64);
    // fibonacci is SELF-recursive with unannotated params: the inferred
    // bounds (PyLe<T> + PyFromInt + PySub<i64, Output = T>) make the
    // integer instantiation compile and run.
    assert_eq!(math_ops::fibonacci(7i64).unwrap(), 13i64);
}

#[test]
fn generic_string_ops_from_rust() {
    // concat_strings(a, b) -> <A as PyAdd<B>>::Output; the literal args
    // are owned so the String instantiation satisfies A: PyAdd<B>.
    assert_eq!(
        string_ops::concat_strings("ab".to_string(), "cd".to_string()).unwrap(),
        "abcd"
    );
    // string_length(s) bounds on Len.
    assert_eq!(string_ops::string_length("hello".to_string()).unwrap(), 5i64);
}
