//! Tests pinning generated-Rust semantics to Python behavior for the
//! correctness fixes: operators, list literals, keyword escaping, assignment
//! mutability, loop else-clauses, with-statements, comprehensions, f-strings,
//! statement separators, await handling, and from-imports.

use python_ast::{parse, CodeGen, CodeGenContext, PythonOptions, SymbolTableScopes};

fn compile(src: &str, name: &str) -> String {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed for {:?}: {}", src, e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            PythonOptions::default(),
            symbols,
        )
        .unwrap_or_else(|e| panic!("codegen failed for {:?}: {}", src, e))
        .to_string()
}

#[test]
fn power_uses_py_pow() {
    let out = compile("y = 2 ** 3", "pow.py");
    assert!(out.contains("py_pow"), "generated: {}", out);
    assert!(!out.contains(". pow"), "generated: {}", out);
}

#[test]
fn power_aug_assign_uses_py_pow() {
    let out = compile("x = 2\nx **= 3", "pow2.py");
    assert!(out.contains("py_pow"), "generated: {}", out);
}

#[test]
fn list_literals_keep_element_types() {
    let out = compile("nums = [1, 2, 3]", "list.py");
    assert!(out.contains("vec ! [1 , 2 , 3]"), "generated: {}", out);
    assert!(!out.contains("to_string"), "generated: {}", out);
}

#[test]
fn rust_keywords_are_escaped() {
    let out = compile("type = 5", "kw.py");
    assert!(out.contains("r#type"), "generated: {}", out);

    let out = compile("def loop():\n    pass\n", "kw2.py");
    assert!(out.contains("fn r#loop"), "generated: {}", out);
}

#[test]
fn assignments_bind_mutably() {
    let out = compile("x = 1", "mut.py");
    assert!(out.contains("let mut x = 1"), "generated: {}", out);
}

#[test]
fn chained_assignment_assigns_each_target() {
    let out = compile("a = b = 1", "chain.py");
    assert!(out.contains("__rython_chain"), "generated: {}", out);
    assert!(out.contains("let mut a"), "generated: {}", out);
    assert!(out.contains("let mut b"), "generated: {}", out);
}

#[test]
fn attribute_assignment_is_not_a_let() {
    let out = compile("def f(obj):\n    obj.field = 1\n", "attr.py");
    assert!(!out.contains("let obj . field"), "generated: {}", out);
    assert!(!out.contains("let mut obj . field"), "generated: {}", out);
}

#[test]
fn for_else_tracks_break() {
    let src = "for x in items:\n    break\nelse:\n    done()\n";
    let out = compile(src, "forelse.py");
    assert!(out.contains("__rython_broke = true"), "generated: {}", out);
    assert!(out.contains("if ! __rython_broke"), "generated: {}", out);
}

#[test]
fn plain_for_has_no_break_flag() {
    let out = compile("for x in items:\n    f(x)\n", "for.py");
    assert!(!out.contains("__rython_broke"), "generated: {}", out);
}

#[test]
fn while_else_tracks_break() {
    let src = "while cond:\n    break\nelse:\n    done()\n";
    let out = compile(src, "whileelse.py");
    assert!(out.contains("__rython_broke = true"), "generated: {}", out);
    assert!(out.contains("if ! __rython_broke"), "generated: {}", out);
}

#[test]
fn nested_loop_break_stays_plain() {
    // The inner loop's break belongs to the inner loop, so the outer
    // for/else needs no flag at all: its else runs unconditionally, and the
    // break stays plain.
    let src = "for x in items:\n    for y in inner:\n        break\nelse:\n    done()\n";
    let out = compile(src, "nested.py");
    assert!(!out.contains("__rython_broke"), "generated: {}", out);
    assert!(out.contains("done ()"), "generated: {}", out);
}

#[test]
fn loop_else_without_break_has_no_flag() {
    // No break in the body: declaring `let mut __rython_broke` would trip
    // deny-warnings builds with unused_mut, so the else runs unconditionally.
    let src = "for x in items:\n    f(x)\nelse:\n    done()\n";
    let out = compile(src, "forelse2.py");
    assert!(!out.contains("__rython_broke"), "generated: {}", out);
    assert!(out.contains("done ()"), "generated: {}", out);
}

#[test]
fn loop_else_break_inside_if_still_tracked() {
    // A break nested in an if still belongs to this loop.
    let src = "for x in items:\n    if x:\n        break\nelse:\n    done()\n";
    let out = compile(src, "forelse3.py");
    assert!(out.contains("__rython_broke = true"), "generated: {}", out);
    assert!(out.contains("if ! __rython_broke"), "generated: {}", out);
}

#[test]
fn with_binds_context_manager() {
    let src = "with open(name) as fh:\n    read(fh)\n";
    let out = compile(src, "with.py");
    assert!(out.contains("let mut fh"), "generated: {}", out);
    assert!(out.contains("open"), "generated: {}", out);
}

#[test]
fn with_without_target_still_evaluates() {
    let src = "with lock():\n    body()\n";
    let out = compile(src, "with2.py");
    assert!(out.contains("let _ = lock ()"), "generated: {}", out);
}

#[test]
fn comprehension_binds_target() {
    let out = compile("doubled = [x * 2 for x in items]", "comp.py");
    assert!(out.contains("for x in"), "generated: {}", out);
    assert!(!out.contains("_item"), "generated: {}", out);
    assert!(out.contains("push"), "generated: {}", out);
}

#[test]
fn comprehension_condition_uses_continue() {
    let out = compile("evens = [x for x in items if x % 2 == 0]", "comp2.py");
    assert!(out.contains("continue"), "generated: {}", out);
}

#[test]
fn multi_generator_comprehension_nests_loops() {
    let out = compile("pairs = [x + y for x in a for y in b]", "comp3.py");
    let for_count = out.matches("for ").count();
    assert!(for_count >= 2, "expected nested loops, generated: {}", out);
    assert!(!out.contains("vec ! []"), "generated: {}", out);
}

#[test]
fn dict_comprehension_inserts_pairs() {
    let out = compile("m = {k: v for k in keys}", "comp4.py");
    assert!(out.contains("insert"), "generated: {}", out);
    assert!(out.contains("HashMap"), "generated: {}", out);
}

#[test]
fn fstring_builds_single_format() {
    let out = compile("s = f\"Hello {name}\"", "fstr.py");
    assert!(out.contains("\"Hello {}\""), "generated: {}", out);
    // No string concatenation with `+`, which didn't even compile.
    assert!(!out.contains("\" + "), "generated: {}", out);
}

#[test]
fn fstring_maps_precision_spec() {
    let out = compile("s = f\"{pi:.2f}\"", "fstr2.py");
    assert!(out.contains("{:.2}"), "generated: {}", out);
}

#[test]
fn fstring_repr_conversion_uses_debug() {
    let out = compile("s = f\"{val!r}\"", "fstr3.py");
    assert!(out.contains("{:?}"), "generated: {}", out);
}

#[test]
fn statements_in_blocks_are_separated() {
    let src = "if cond:\n    first()\n    second()\n";
    let out = compile(src, "sep.py");
    let first = out.find("first ()").expect("first call present");
    let second = out.find("second ()").expect("second call present");
    let between = &out[first..second];
    assert!(between.contains(';'), "no separator between calls: {}", out);
}

#[test]
fn async_calls_do_not_guess_await() {
    let src = "async def f(x):\n    return abs(x)\n";
    let out = compile(src, "await.py");
    assert!(!out.contains(". await"), "generated: {}", out);
}

#[test]
fn explicit_await_still_awaits() {
    let src = "async def f(x):\n    return await g(x)\n";
    let out = compile(src, "await2.py");
    assert!(out.contains(". await"), "generated: {}", out);
}

#[test]
fn from_import_brings_name_into_scope() {
    let out = compile("from os import path", "imp.py");
    assert!(out.contains("use stdpython :: os :: path ;"), "generated: {}", out);
}

#[test]
fn from_import_with_alias() {
    let out = compile("from os import path as p", "imp2.py");
    assert!(out.contains("use stdpython :: os :: path as p ;"), "generated: {}", out);
}

#[test]
fn lambda_parameters_are_bare_names() {
    let out = compile("f = lambda x: x", "lam.py");
    assert!(out.contains("| x |"), "generated: {}", out);
    assert!(!out.contains("impl Into"), "generated: {}", out);
}

#[test]
fn return_type_inferred_from_int_constant() {
    let out = compile("def f():\n    return 42\n", "ret.py");
    assert!(out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn return_type_inferred_from_fstring() {
    let out = compile("def f():\n    return f\"x={x}\"\n", "ret2.py");
    assert!(out.contains("-> String"), "generated: {}", out);
}

#[test]
fn return_type_inferred_from_string_literal() {
    let out = compile("def f():\n    return \"hi\"\n", "ret3.py");
    assert!(out.contains("-> & 'static str"), "generated: {}", out);
}

#[test]
fn mixed_returns_get_no_annotation() {
    let out = compile("def f(c):\n    if c:\n        return 1\n    return \"s\"\n", "ret4.py");
    assert!(!out.contains("->"), "generated: {}", out);
}

#[test]
fn bare_return_gets_no_annotation() {
    let out = compile("def f():\n    return\n", "ret5.py");
    assert!(!out.contains("->"), "generated: {}", out);
}

#[test]
fn return_type_inferred_through_local_variable() {
    let out = compile("def f():\n    n = 5\n    n -= 1\n    return n\n", "ret6.py");
    assert!(out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn partial_return_gets_no_annotation() {
    // The fall-through path implicitly returns None, so annotating -> i64
    // would make the generated fn fail to compile.
    let out = compile("def f(c):\n    if c:\n        return 1\n", "ret7.py");
    assert!(!out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn return_in_loop_only_gets_no_annotation() {
    let out = compile("def f(items):\n    for x in items:\n        return 1\n", "ret8.py");
    assert!(!out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn exhaustive_if_else_returns_get_annotation() {
    let src = "def f(c):\n    if c:\n        return 1\n    else:\n        return 2\n";
    let out = compile(src, "ret9.py");
    assert!(out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn annotated_parameters_map_to_rust_types() {
    let out = compile("def f(a: int, b: float, c: str, d: bool):\n    pass\n", "ann_params.py");
    assert!(out.contains("a : i64"), "generated: {}", out);
    assert!(out.contains("b : f64"), "generated: {}", out);
    assert!(out.contains("c : String"), "generated: {}", out);
    assert!(out.contains("d : bool"), "generated: {}", out);
    assert!(!out.contains(": int"), "generated: {}", out);
}

#[test]
fn return_annotation_used_when_inference_fails() {
    let out = compile("def f(x: int) -> int:\n    return x + 1\n", "ann_ret.py");
    assert!(out.contains("-> i64"), "generated: {}", out);
}

#[test]
fn string_repetition_uses_multiply_string() {
    let out = compile("s = \"!\" * 3", "strmul.py");
    assert!(out.contains("multiply_string"), "generated: {}", out);
    let out = compile("s = 3 * \"!\"", "strmul2.py");
    assert!(out.contains("multiply_string"), "generated: {}", out);
    // Numeric multiplication is untouched.
    let out = compile("n = 3 * 4", "nummul.py");
    assert!(!out.contains("multiply_string"), "generated: {}", out);
}

#[test]
fn stdlib_from_import_anchors_to_stdpython() {
    let out = compile("from os import path", "imp3.py");
    assert!(out.contains("use stdpython :: os :: path ;"), "generated: {}", out);
}

#[test]
fn sibling_from_import_anchors_to_crate() {
    let out = compile("from helpers import util", "imp4.py");
    assert!(out.contains("use crate :: helpers :: util ;"), "generated: {}", out);
}

#[test]
fn defaulted_annotated_parameter_maps_type() {
    // Defaulted parameters lower to plain required parameters with mapped
    // types (never the raw Python name, and no Option wrapper, which
    // type-checked against neither bodies nor call sites).
    let out = compile("def f(x: int = 0):\n    return x\n", "def_param.py");
    assert!(out.contains("x : i64"), "generated: {}", out);
    assert!(!out.contains("Option"), "generated: {}", out);
    assert!(!out.contains(": int"), "generated: {}", out);
}

#[test]
fn kwonly_annotated_parameter_maps_type() {
    let out = compile("def f(*, x: int):\n    pass\n", "kwonly.py");
    assert!(out.contains("x : i64"), "generated: {}", out);
    assert!(!out.contains(": int"), "generated: {}", out);
}

#[test]
fn annotation_ignored_when_body_can_fall_through() {
    // A return annotation must not be applied when a path can reach the end
    // of the function without returning (the implicit tail is `()`) — but
    // ignoring it is a lossy conversion that likely marks a source bug, so
    // the generated function must carry a warning note saying so.
    let out = compile("def f(c) -> int:\n    if c:\n        return 1\n", "ann_partial.py");
    assert!(!out.contains("-> i64"), "generated: {}", out);
    assert!(out.contains("deprecated"), "generated: {}", out);
    assert!(
        out.contains("return annotation was ignored")
            || out.contains("return annotation `-> int`")
            || out.contains("`-> int` return annotation"),
        "warning note should name the ignored annotation: {}",
        out
    );

    // A function that honors its annotation carries no warning.
    let out = compile("def g() -> int:\n    return 1\n", "ann_honored.py");
    assert!(!out.contains("deprecated"), "generated: {}", out);

    // `-> None` on a fall-through body is accurate, not lossy.
    let out = compile("def h() -> None:\n    print(1)\n", "ann_none.py");
    assert!(!out.contains("deprecated"), "generated: {}", out);
}

#[test]
fn multiple_lossy_conversions_fold_into_one_attribute() {
    // Rust allows only one #[deprecated] per item, so a function with both a
    // dropped default and an ignored return annotation must fold both notes
    // into a single attribute.
    let out = compile(
        "def f(c, x: int = 3) -> int:\n    if c:\n        return x\n",
        "lossy_both.py",
    );
    assert_eq!(
        out.matches("deprecated").count(),
        1,
        "exactly one #[deprecated] attribute: {}",
        out
    );
    assert!(out.contains("were dropped"), "generated: {}", out);
    assert!(out.contains("return annotation"), "generated: {}", out);
}

#[test]
fn lossy_warnings_can_be_suppressed_by_options() {
    let src = "def f(x: int = 3) -> int:\n    if x:\n        return x\n";
    let module = parse(src, "suppress.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = PythonOptions {
        lossy_warnings: false,
        ..Default::default()
    };
    let out = module
        .to_rust(CodeGenContext::Module("suppress".into()), options, symbols)
        .unwrap()
        .to_string();
    assert!(!out.contains("deprecated"), "generated: {}", out);
}

#[test]
fn dropped_defaults_emit_call_site_warning() {
    // Dropping a Python default is a semantic change; the generated function
    // must carry a #[deprecated] note so consumer call sites are warned.
    let out = compile("def f(x: int = 3) -> int:\n    return x\n", "warn_def.py");
    assert!(out.contains("deprecated"), "generated: {}", out);
    assert!(out.contains("were dropped"), "generated: {}", out);

    // No defaults, no warning attribute.
    let out = compile("def g(x: int) -> int:\n    return x\n", "no_warn.py");
    assert!(!out.contains("deprecated"), "generated: {}", out);
}
