//! python_module! integration test (issue #456 / Devin review on #460):
//! a standalone macro conversion that IMPORTS A SIBLING.
//!
//! The macro converts one module with no index of its siblings, so the
//! importer's `import sibling_helper` must still lower to
//! `use crate::sibling_helper`, and `sibling_helper.greet()` must stay a
//! real call. When the external-module classification was applied to macro
//! conversions, the import was dropped and the call became a boxed `None`,
//! silently returning the wrong answer.
use python_mod::python_module;

python_module!(sibling_helper);
python_module!(sibling_caller);

#[test]
fn a_macro_conversion_calls_through_to_its_sibling() {
    assert_eq!(
        sibling_helper::greet().unwrap(),
        "hello from sibling".to_string()
    );
    assert_eq!(sibling_caller::call_greet().unwrap(), "hello from sibling");
    assert_eq!(sibling_caller::call_add(2i64, 3i64).unwrap(), 5i64);
}
