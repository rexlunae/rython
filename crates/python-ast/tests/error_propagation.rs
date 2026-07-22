//! Tests pinning the error-propagation contract: unsupported or unconvertible
//! Python must surface as structured `Err` values — never as panics — and the
//! errors must carry enough location information to point at the user's code.

use python_ast::{parse, parse_enhanced, CodeGen, CodeGenContext, PythonOptions, SymbolTableScopes};

/// Statements rython does not support must produce an error, not a panic.
#[test]
fn unsupported_statement_returns_err() {
    let result = parse("match x:\n    case 1:\n        pass", "match_test.py");
    let err = result.expect_err("match statement should be rejected");
    let message = err.to_string();
    assert!(
        message.contains("Match") || message.contains("not yet supported"),
        "error should name the unsupported construct: {}",
        message
    );
}

/// `del` is unsupported; it must error with the statement name and position.
#[test]
fn del_statement_returns_err_with_location() {
    let result = parse("x = 1\ndel x", "del_test.py");
    let err = result.expect_err("del statement should be rejected");
    let message = err.to_string();
    assert!(
        message.contains("Delete"),
        "error should name the construct: {}",
        message
    );
    assert!(
        message.contains("line 2"),
        "error should carry the source line: {}",
        message
    );
}

/// Slice subscripts are not yet representable; they must error, not panic.
#[test]
fn slice_subscript_returns_err() {
    let result = parse("a[1:3]", "slice_test.py");
    assert!(result.is_err(), "slice subscript should be rejected");
}

/// Syntax errors keep flowing through the enhanced parser with location and
/// help fields intact.
#[test]
fn syntax_error_carries_structured_fields() {
    let err = parse_enhanced("def broken(", "broken.py").expect_err("syntax error");
    assert!(err.get_field("location").is_some(), "location field");
    assert!(err.get_field("help").is_some(), "help field");
    assert!(
        err.downcast_ref::<python_ast::ParseError>().is_some(),
        "typed payload should be recoverable"
    );
}

/// Codegen failures propagate as Err from Module::to_rust (instead of the old
/// behavior of panicking inside the statement walker), and the error names the
/// module's source file.
#[test]
fn codegen_error_propagates_with_filename() {
    // `x @ y` (MatMult) parses fine but has no codegen implementation.
    let module = parse("q = a @ b", "matmul_test.py").expect("parses");
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let result = module.to_rust(
        CodeGenContext::Module("matmul_test".to_string()),
        PythonOptions::default(),
        symbols,
    );
    let err = result.expect_err("MatMult codegen should fail");
    let chain = python_ast::format_error_chain(err.as_ref());
    assert!(
        chain.contains("matmul_test.py"),
        "error should name the source file: {}",
        chain
    );
}

/// The happy path still works end to end.
#[test]
fn supported_module_still_compiles() {
    let module = parse("def add(a, b):\n    return a + b\n", "ok.py").expect("parses");
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let rust = module
        .to_rust(
            CodeGenContext::Module("ok".to_string()),
            PythonOptions::default(),
            symbols,
        )
        .expect("codegen succeeds");
    assert!(rust.to_string().contains("fn add"));
}
