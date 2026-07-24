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
fn assignments_hoist_declaration_and_store() {
    // Assigned names are hoisted to a declaration and each assignment is a
    // plain store (a `let mut` per assignment would shadow inside nested
    // blocks instead of assigning). A single store needs no `mut`.
    // (A literal here would become a module constant static instead, so
    // use a computed value.)
    let out = compile("x = 1 + 1", "mut.py");
    assert!(out.contains("let x"), "generated: {}", out);
    assert!(!out.contains("let mut x"), "single store needs no mut: {}", out);
}

#[test]
fn mut_is_inferred_only_where_needed() {
    // Branch-exclusive initialization: no path assigns twice, so no mut —
    // rustc would warn unused_mut otherwise.
    let src = "def f(c) -> int:\n    if c:\n        x = 1\n    else:\n        x = 2\n    return x\n";
    let out = compile(src, "branches.py");
    assert!(out.contains("let x ;"), "generated: {}", out);
    assert!(!out.contains("let mut x"), "generated: {}", out);

    // A store inside a loop may execute repeatedly: mut required.
    let src = "def g(items):\n    total = 0\n    for i in items:\n        total = total + i\n    return total\n";
    let out = compile(src, "loopmut.py");
    assert!(out.contains("let mut total"), "generated: {}", out);

    // A mutating method call requires a mutable binding.
    let out = compile("def h():\n    items = []\n    items.append(1)\n", "append.py");
    assert!(out.contains("let mut items"), "generated: {}", out);

    // A parameter that is only read is not rebound.
    let out = compile("def k(n: int) -> int:\n    return n\n", "readonly.py");
    assert!(!out.contains("let mut n"), "generated: {}", out);
}

#[test]
fn nested_block_assignment_stores_into_the_outer_variable() {
    // `x = 2` inside the if must update the function-scoped x, not create a
    // shadowing binding that dies at the end of the block.
    let src = "def pick(c) -> int:\n    x = 1\n    if c:\n        x = 2\n    return x\n";
    let out = compile(src, "scope.py");
    assert_eq!(
        out.matches("let mut x").count(),
        1,
        "one declaration, plain stores elsewhere: {}",
        out
    );
    assert!(
        out.contains("if (c) . is_truthy () { x = 2"),
        "generated: {}",
        out
    );
}

#[test]
fn assigned_parameters_are_rebound_mutably() {
    // Rust parameters are immutable; a parameter the body assigns to is
    // rebound as a mutable local first.
    let out = compile("def f(n: int) -> int:\n    n = n + 1\n    return n\n", "param.py");
    assert!(out.contains("let mut n = n"), "generated: {}", out);
}

#[test]
fn chained_assignment_assigns_each_target() {
    let out = compile("a = b = 1", "chain.py");
    assert!(out.contains("__rython_chain"), "generated: {}", out);
    assert!(out.contains("let a"), "generated: {}", out);
    assert!(out.contains("let b"), "generated: {}", out);
    assert!(out.contains("a = __rython_chain"), "generated: {}", out);
    assert!(out.contains("b = __rython_chain"), "generated: {}", out);
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
    assert!(out.contains("PyDict"), "generated: {}", out);
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
    assert!(out.contains("-> Result < i64 , PyException >"), "generated: {}", out);
}

#[test]
fn return_type_inferred_from_fstring() {
    let out = compile("def f():\n    return f\"x={x}\"\n", "ret2.py");
    assert!(out.contains("-> Result < String , PyException >"), "generated: {}", out);
}

#[test]
fn return_type_inferred_from_string_literal() {
    let out = compile("def f():\n    return \"hi\"\n", "ret3.py");
    assert!(out.contains("-> Result < & 'static str , PyException >"), "generated: {}", out);
}

#[test]
fn mixed_returns_get_no_annotation() {
    let out = compile("def f(c):\n    if c:\n        return 1\n    return \"s\"\n", "ret4.py");
    assert!(out.contains("-> Result < () , PyException >"), "generated: {}", out);
}

#[test]
fn bare_return_gets_no_annotation() {
    let out = compile("def f():\n    return\n", "ret5.py");
    assert!(out.contains("-> Result < () , PyException >"), "generated: {}", out);
    assert!(out.contains("return Ok (())"), "generated: {}", out);
}

#[test]
fn return_type_inferred_through_local_variable() {
    let out = compile("def f():\n    n = 5\n    n -= 1\n    return n\n", "ret6.py");
    assert!(out.contains("-> Result < i64 , PyException >"), "generated: {}", out);
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
    assert!(out.contains("-> Result < i64 , PyException >"), "generated: {}", out);
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
    assert!(out.contains("-> Result < i64 , PyException >"), "generated: {}", out);
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
fn try_except_lowers_to_result_handling() {
    let src = concat!(
        "def f(n):\n",
        "    try:\n",
        "        raise ValueError(\"bad\")\n",
        "    except ValueError as e:\n",
        "        print(e)\n",
        "    except (TypeError, KeyError):\n",
        "        print(\"other\")\n",
    );
    let out = compile(src, "try.py");
    // The body runs in a closure returning Result<(), PyException>.
    assert!(
        out.contains("Result < () , PyException >"),
        "generated: {}",
        out
    );
    // raise inside the try returns an Err the handlers can match.
    assert!(
        out.contains("return Err (PyException :: new (\"ValueError\""),
        "generated: {}",
        out
    );
    // Handlers are guard-matched arms, in order; the tuple form ORs.
    assert!(
        out.contains("if __rython_exc . matches (\"ValueError\")"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("matches (\"TypeError\") || __rython_exc . matches (\"KeyError\")"),
        "generated: {}",
        out
    );
    // `as e` binds the caught exception.
    assert!(out.contains("let mut e = __rython_exc . clone ()"), "generated: {}", out);
    // An unmatched exception re-raises as an Err out of the function.
    assert!(
        out.contains("Err (__rython_exc) => { return Err (__rython_exc) ; }"),
        "generated: {}",
        out
    );
}

#[test]
fn try_handler_bodies_only_run_on_matching_error() {
    // The old lowering ran every handler body unconditionally after the try
    // body; the handler statements must now live inside match arms.
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        work()\n",
        "    except Exception:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "tryarm.py");
    let arm_pos = out.find("Err (__rython_exc)").expect("handler arm");
    let cleanup_pos = out.find("cleanup ()").expect("handler body");
    assert!(
        cleanup_pos > arm_pos,
        "handler body must be inside the Err arm: {}",
        out
    );
}

#[test]
fn nested_raise_propagates_to_outer_try() {
    // A try inside a try: the inner unmatched arm returns Err out of the
    // *outer* closure instead of panicking.
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        try:\n",
        "            raise KeyError(\"k\")\n",
        "        except ValueError:\n",
        "            pass\n",
        "    except KeyError:\n",
        "        pass\n",
    );
    let out = compile(src, "nested_try.py");
    assert!(
        out.contains("Err (__rython_exc) => { return Err (__rython_exc) ; }"),
        "inner unmatched exception must propagate as Err: {}",
        out
    );
}

#[test]
fn finally_runs_before_reraise() {
    let src = concat!(
        "def f():\n",
        "    try:\n",
        "        work()\n",
        "    except ValueError:\n",
        "        pass\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "finally.py");
    // finally body appears both after the match (normal paths) and in the
    // unmatched-reraise arm (before propagation).
    assert!(out.matches("cleanup ()").count() >= 2, "generated: {}", out);
}

#[test]
fn finally_runs_before_handler_and_else_returns() {
    // Python: finally always executes before control leaves the try
    // statement — including when an except handler or else clause returns
    // or raises. Handler/else bodies must route through the finally, not
    // return straight out of the function.
    let src = concat!(
        "def f(n: int) -> int:\n",
        "    try:\n",
        "        check(n)\n",
        "    except ValueError:\n",
        "        return 0\n",
        "    else:\n",
        "        return 1\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "finally_handler.py");
    // Both the handler return and the else return thread out through a
    // ControlFlow closure whose Break arm runs cleanup() first.
    assert_eq!(
        out.matches("Ok (std :: ops :: ControlFlow :: Break (__rython_ret)) => { cleanup () ; return Ok (__rython_ret) ; }")
            .count(),
        2,
        "handler and else returns must run the finally first: {}",
        out
    );

    // A raise inside a handler also runs the finally before propagating.
    let src = concat!(
        "def g(n: int):\n",
        "    try:\n",
        "        check(n)\n",
        "    except ValueError:\n",
        "        raise RuntimeError(\"rethrown\")\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "finally_reraise.py");
    assert!(
        out.contains("Err (__rython_reraise) => { cleanup () ; return Err (__rython_reraise) ; }"),
        "handler raise must run the finally first: {}",
        out
    );

    // Without a finally clause, handler bodies stay inline — no closure.
    let src = concat!(
        "def h(n: int) -> int:\n",
        "    try:\n",
        "        check(n)\n",
        "    except ValueError:\n",
        "        return 0\n",
        "    return 1\n",
    );
    let out = compile(src, "no_finally.py");
    assert!(!out.contains("__rython_inner"), "generated: {}", out);
}

#[test]
fn awaited_async_calls_propagate_exceptions() {
    // Async functions register in the symbol table like ordinary ones, so
    // calls to them get `?` — reordered after `.await` so it unwraps the
    // awaited Result, not the future.
    let src = concat!(
        "async def helper() -> int:\n",
        "    return 1\n",
        "\n",
        "async def caller() -> int:\n",
        "    return await helper()\n",
    );
    let out = compile(src, "async_prop.py");
    assert!(
        out.contains("helper () . await ?"),
        "awaited user call must unwrap the Result: {}",
        out
    );
}

#[test]
fn bare_trailing_return_gets_no_unreachable_tail() {
    // A bare `return` fully exits the function (it extracts as returning
    // None), so no Ok(()) tail may follow it — that would be unreachable
    // code, tripping deny-warnings builds.
    let out = compile("def f():\n    work()\n    return\n", "bareret.py");
    assert!(out.contains("return Ok (())"), "generated: {}", out);
    assert!(
        !out.contains("return Ok (()) ; Ok (())"),
        "no unreachable tail after a trailing bare return: {}",
        out
    );
}

#[test]
fn raise_returns_err_from_the_function() {
    // Functions return Result<T, PyException>, so raising anywhere is
    // returning Err — callers propagate it with `?`, as Python propagates
    // exceptions up the call stack.
    let out = compile(
        "def f():\n    raise RuntimeError(\"boom\")\n",
        "raise.py",
    );
    assert!(
        out.contains("return Err (PyException :: new (\"RuntimeError\""),
        "generated: {}",
        out
    );
    assert!(!out.contains("panic !"), "generated: {}", out);
}

#[test]
fn calls_to_user_functions_propagate_with_question_mark() {
    let src = concat!(
        "def helper() -> int:\n",
        "    return 1\n",
        "\n",
        "def caller() -> int:\n",
        "    return helper() + 1\n",
    );
    let out = compile(src, "prop.py");
    assert!(out.contains("helper () ?"), "generated: {}", out);

    // Builtins that don't raise stay plain.
    let out = compile("def f(x: int):\n    print(x)\n", "plaincall.py");
    assert!(out.contains("print (x)"), "generated: {}", out);
    assert!(!out.contains("print (x) ?"), "generated: {}", out);
}

#[test]
fn return_inside_try_threads_through_controlflow() {
    // A return in a try body must escape the closure, run the finally, and
    // return from the function.
    let src = concat!(
        "def f(n: int) -> int:\n",
        "    try:\n",
        "        return n\n",
        "    except ValueError:\n",
        "        return 0\n",
        "    finally:\n",
        "        cleanup()\n",
    );
    let out = compile(src, "trystmt_ret.py");
    assert!(
        out.contains("ControlFlow :: Break (n)"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("Ok (std :: ops :: ControlFlow :: Break (__rython_ret)) => { cleanup () ; return Ok (__rython_ret) ; }"),
        "finally must run before the returned value leaves: {}",
        out
    );
}

#[test]
fn assert_lowers_to_assertion_error() {
    let out = compile("def f(n):\n    assert n > 0, \"need positive\"\n", "assert.py");
    assert!(out.contains("if ! ((n) > (0))"), "generated: {}", out);
    assert!(
        out.contains("PyException :: new (\"AssertionError\""),
        "generated: {}",
        out
    );

    // Inside a try, a failed assert is catchable.
    let src = concat!(
        "def f(n):\n",
        "    try:\n",
        "        assert n > 0\n",
        "    except AssertionError:\n",
        "        pass\n",
    );
    let out = compile(src, "assert_try.py");
    assert!(
        out.contains("return Err (PyException :: new (\"AssertionError\""),
        "generated: {}",
        out
    );
}

#[test]
fn unary_plus_emits_no_invalid_operator() {
    // Rust has no unary +; `+x` is the identity.
    let out = compile("y = +x", "uadd.py");
    assert!(!out.contains("= + x"), "generated: {}", out);
    assert!(out.contains("y = (x)"), "generated: {}", out);
}

#[test]
fn conditions_apply_python_truthiness() {
    // Non-bool condition: wrapped in is_truthy (empty string/list and zero
    // are false, as in Python).
    let out = compile("def f(items):\n    if items:\n        work()\n", "truthy.py");
    assert!(out.contains("if (items) . is_truthy ()"), "generated: {}", out);

    let out = compile("def f(n):\n    while n:\n        work()\n", "truthy_while.py");
    assert!(out.contains("while (n) . is_truthy ()"), "generated: {}", out);

    // Comparisons already yield bool: no wrapping.
    let out = compile("def f(n: int):\n    if n < 0:\n        work()\n", "truthy_cmp.py");
    assert!(!out.contains("is_truthy"), "generated: {}", out);

    // Boolean operators recurse into operands; `not` negates a condition.
    let out = compile("def f(a, b):\n    if a and not b:\n        work()\n", "truthy_bool.py");
    assert!(
        out.contains("((a) . is_truthy ()) && (! ((b) . is_truthy ()))"),
        "generated: {}",
        out
    );
}

#[test]
fn is_none_lowers_to_py_is_none() {
    let out = compile("def f(x):\n    if x is None:\n        work()\n", "isnone.py");
    assert!(out.contains("(x) . py_is_none ()"), "generated: {}", out);

    let out = compile("def f(x):\n    if x is not None:\n        work()\n", "isnotnone.py");
    assert!(out.contains("! (x) . py_is_none ()"), "generated: {}", out);

    // `is` between two non-None values keeps the identity approximation.
    let out = compile("found = a is b", "isplain.py");
    assert!(out.contains("& a == & b"), "generated: {}", out);
}

#[test]
fn python_list_methods_map_to_correct_rust() {
    let src = concat!(
        "def f() -> int:\n",
        "    items = [1, 2, 3]\n",
        "    items.append(4)\n",
        "    items.remove(2)\n",
        "    items.insert(0, 9)\n",
        "    last = items.pop()\n",
        "    return last + items.count(9)\n",
    );
    let out = compile(src, "listops.py");
    // append pushes one element (Vec::append concatenates — wrong).
    assert!(out.contains("(items) . push (4)"), "generated: {}", out);
    // remove removes by value and raises ValueError when absent.
    assert!(out.contains("position"), "generated: {}", out);
    assert!(out.contains("\"ValueError\""), "generated: {}", out);
    // insert applies Python index rules (negatives, clamping) via py_insert.
    assert!(out.contains("py_insert (0 , 9)"), "generated: {}", out);
    // pop raises a catchable IndexError instead of returning an Option.
    assert!(out.contains("\"IndexError\""), "generated: {}", out);
    assert!(out.contains("pop () . ok_or_else"), "generated: {}", out);
    // count passes by reference to the PyListOps method.
    assert!(out.contains("count (& (9))"), "generated: {}", out);
}

#[test]
fn python_str_methods_map_through_pystrops() {
    let src = concat!(
        "def f(s: str) -> str:\n",
        "    parts = s.split()\n",
        "    head = s.split(\",\")\n",
        "    n = s.find(\"x\")\n",
        "    return \"-\".join(parts)\n",
    );
    let out = compile(src, "strops.py");
    assert!(out.contains("py_split_whitespace ()"), "generated: {}", out);
    assert!(out.contains("py_split (& (\",\")) ?"), "generated: {}", out);
    assert!(out.contains("py_find (& (\"x\"))"), "generated: {}", out);
    assert!(out.contains(". join (parts)"), "generated: {}", out);
}

#[test]
fn str_parameters_accept_borrowed_and_owned_strings() {
    let out = compile("def shout(name: str) -> str:\n    return name.upper()\n", "strparam.py");
    // The parameter is generic over Into<String>, converted once up front.
    assert!(
        out.contains("name : impl Into < String >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("let name : String = name . into ()"),
        "generated: {}",
        out
    );
}

#[test]
fn subscripts_lower_through_py_index() {
    // Reads follow Python index rules (negatives, catchable IndexError).
    let out = compile("def f(items: list[int], i: int) -> int:\n    return items[i]\n", "sub.py");
    assert!(out.contains("(items) . py_index (i) ?"), "generated: {}", out);

    // Stores go through py_set_index, not the Load lowering.
    let out = compile(
        "def f(items: list[int]):\n    items[0] = 5\n",
        "substore.py",
    );
    assert!(
        out.contains("(items) . py_set_index (0 , 5) ?"),
        "generated: {}",
        out
    );
    assert!(!out.contains("py_index (0) ? ="), "generated: {}", out);

    // Dict stores insert; catchable KeyError on reads comes from PyIndex.
    let out = compile("def f():\n    d = {\"a\": 1}\n    d[\"b\"] = 2\n    return d[\"a\"]\n", "dictsub.py");
    assert!(out.contains("py_set_index (\"b\" , 2) ?"), "generated: {}", out);
    assert!(out.contains("py_index (\"a\") ?"), "generated: {}", out);
}

#[test]
fn slices_lower_through_py_slice() {
    let out = compile("def f(items: list[int]):\n    return items[1:3]\n", "slice1.py");
    assert!(
        out.contains("py_slice (Some (1) , Some (3) , None)"),
        "generated: {}",
        out
    );

    let out = compile("def f(s: str) -> str:\n    return s[::-1]\n", "slice2.py");
    assert!(
        out.contains("py_slice (None , None , Some (- 1))"),
        "generated: {}",
        out
    );
}

#[test]
fn container_annotations_map_to_rust_types() {
    let out = compile("def f(a: list[int], b: dict[str, int], c: set[int]):\n    pass\n", "generics.py");
    assert!(out.contains("a : Vec < i64 >"), "generated: {}", out);
    assert!(
        out.contains("b : PyDict < String , i64 >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("c : std :: collections :: HashSet < i64 >"),
        "generated: {}",
        out
    );
}

#[test]
fn augmented_assignment_to_subscript_reads_and_stores() {
    // counts[k] += 1 is read-modify-write through py_index/py_set_index —
    // the Load lowering yields a temporary, not a place.
    let out = compile(
        "def f():\n    counts = {\"a\": 1}\n    counts[\"a\"] += 5\n",
        "augsub.py",
    );
    assert!(
        out.contains("py_index (__rython_idx . clone ()) ?"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("py_set_index (__rython_idx , (__rython_elem) . py_add (& (5))) ?"),
        "generated: {}",
        out
    );

    // Other operators combine with the read value too.
    let out = compile(
        "def f():\n    nums = [1, 2]\n    nums[-1] *= 2\n",
        "augsub2.py",
    );
    assert!(
        out.contains("py_set_index (__rython_idx , __rython_elem * 2) ?"),
        "generated: {}",
        out
    );
}

#[test]
fn bare_numeric_literals_are_anchored_in_addition() {
    // `1 + 2` with no type anchor: the PyAdd receiver must have a concrete
    // type, or trait resolution fails before integer-literal fallback.
    let out = compile("y = 1 + 2", "anchor.py");
    assert!(
        out.contains("((1) as i64) . py_add (& ((2) as i64))"),
        "generated: {}",
        out
    );

    let out = compile("y = 1.5 + 2.5", "anchor2.py");
    assert!(
        out.contains("((1.5) as f64) . py_add"),
        "generated: {}",
        out
    );
}

#[test]
fn addition_lowers_through_py_add() {
    // Python + covers String + String and list concat, which Rust's Add
    // doesn't; operands are borrowed so variables stay usable.
    let out = compile("def f(a: str, b: str) -> str:\n    return a + b\n", "addstr.py");
    assert!(out.contains("(a) . py_add (& (b))"), "generated: {}", out);

    let out = compile("def f(n: int) -> int:\n    n += 1\n    return n\n", "addaug.py");
    assert!(out.contains("n = (n) . py_add (& (1))"), "generated: {}", out);
}

#[test]
fn dict_literals_and_methods_lower_through_pydict() {
    // Dict literals are insertion-ordered PyDicts, not HashMaps.
    let out = compile("d = {\"a\": 1}", "dictlit.py");
    assert!(out.contains("PyDict :: from"), "generated: {}", out);
    assert!(!out.contains("HashMap :: from"), "generated: {}", out);

    // Method mappings: get/pop/setdefault/views.
    let src = concat!(
        "def f() -> int:\n",
        "    d = {\"a\": 1}\n",
        "    x = d.get(\"a\", 0)\n",
        "    y = d.pop(\"a\")\n",
        "    z = d.pop(\"gone\", 9)\n",
        "    d.setdefault(\"b\", 2)\n",
        "    ks = d.keys()\n",
        "    vs = d.values()\n",
        "    it = d.items()\n",
        "    return x + y + z\n",
    );
    let out = compile(src, "dictops.py");
    assert!(out.contains("py_get_default (& (\"a\") , 0)"), "generated: {}", out);
    assert!(out.contains("py_pop (\"a\") ?"), "generated: {}", out);
    assert!(out.contains("py_pop_default (\"gone\" , 9)"), "generated: {}", out);
    assert!(out.contains("py_setdefault (\"b\" , 2)"), "generated: {}", out);
    assert!(out.contains("py_keys ()"), "generated: {}", out);
    assert!(out.contains("py_values ()"), "generated: {}", out);
    assert!(out.contains("py_items ()"), "generated: {}", out);

    // get with one argument returns an Option (value-or-None).
    let out = compile("def g(d: dict[str, int]):\n    v = d.get(\"k\")\n", "dictget.py");
    assert!(out.contains("py_get (& (\"k\"))"), "generated: {}", out);
}

#[test]
fn keyword_arguments_map_to_parameter_positions() {
    let src = concat!(
        "def volume(w: int, h: int, d: int) -> int:\n",
        "    return w * h * d\n",
        "\n",
        "def f() -> int:\n",
        "    return volume(d=2, w=3, h=4)\n",
    );
    let out = compile(src, "kw.py");
    // Keywords land in signature order regardless of call order.
    assert!(out.contains("volume (3 , 4 , 2) ?"), "generated: {}", out);
}

#[test]
fn omitted_defaults_fill_at_the_call_site() {
    let src = concat!(
        "def greet(name: str = \"world\", excited: bool = False) -> str:\n",
        "    return name\n",
        "\n",
        "def f() -> str:\n",
        "    return greet()\n",
        "\n",
        "def g() -> str:\n",
        "    return greet(excited=True)\n",
    );
    let out = compile(src, "kwdef.py");
    assert!(
        out.contains("greet (\"world\" , false) ?"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("greet (\"world\" , true) ?"),
        "keyword for the second param leaves the first defaulted: {}",
        out
    );
}

#[test]
fn keywords_on_unknown_callees_error_loudly() {
    // Without a signature the keyword order can't be checked — refusing
    // beats silently reordering arguments.
    let module = parse("unknown_func(a=1)\n", "kwunknown.py").unwrap();
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let err = module
        .to_rust(
            CodeGenContext::Module("kwunknown".into()),
            PythonOptions::default(),
            symbols,
        )
        .expect_err("keywords on unknown callee must not convert");
    assert!(
        format!("{}", err).contains("signature"),
        "error: {}",
        err
    );
}

#[test]
fn dict_comprehensions_build_ordered_pydicts() {
    // Comprehension-built dicts preserve insertion order like literals.
    let out = compile(
        "def f(items: list[int]):\n    return {x: x * 2 for x in items}\n",
        "dictcomp.py",
    );
    assert!(out.contains("PyDict :: new ()"), "generated: {}", out);
    assert!(!out.contains("HashMap :: new ()"), "generated: {}", out);
}

#[test]
fn none_lowers_to_option() {
    // x = None initializes an Option; later non-None stores wrap in Some
    // so both arms unify to Option<T>.
    let src = concat!(
        "def f(items: list[int]) -> int:\n",
        "    found = None\n",
        "    for x in items:\n",
        "        found = x\n",
        "    if found is None:\n",
        "        return -1\n",
        "    return 0\n",
    );
    let out = compile(src, "opt.py");
    assert!(out.contains("found = None"), "generated: {}", out);
    assert!(out.contains("found = Some (x)"), "generated: {}", out);
    assert!(out.contains("(found) . py_is_none ()"), "generated: {}", out);
}

#[test]
fn optional_annotations_map_to_option() {
    let out = compile(
        "def f(tag: Optional[int], n: int | None) -> int:\n    return 0\n",
        "optann.py",
    );
    assert!(out.contains("tag : Option < i64 >"), "generated: {}", out);
    assert!(out.contains("n : Option < i64 >"), "generated: {}", out);
}

#[test]
fn optional_parameters_wrap_arguments_at_call_sites() {
    let src = concat!(
        "def label(tag: Optional[int]) -> int:\n",
        "    return 0\n",
        "\n",
        "def f() -> int:\n",
        "    a = label(7)\n",
        "    b = label(None)\n",
        "    return a + b\n",
    );
    let out = compile(src, "optcall.py");
    assert!(out.contains("label (Some (7)) ?"), "generated: {}", out);
    assert!(out.contains("label (None) ?"), "generated: {}", out);
}

#[test]
fn optional_stores_from_option_values_do_not_double_wrap() {
    // The RHS already yields an Option (dict.get, another optional name, an
    // Optional-returning call): wrapping it again would bury an absent value
    // as Some(None) and flip a later `is None` check.
    let src = concat!(
        "def probe(d: dict[str, int], keys: list[str]) -> int:\n",
        "    result = None\n",
        "    for k in keys:\n",
        "        result = d.get(k)\n",
        "    alias = None\n",
        "    alias = result\n",
        "    if alias is None:\n",
        "        return -1\n",
        "    return 0\n",
    );
    let out = compile(src, "optget.py");
    assert!(
        out.contains("result = (d) . py_get"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("Some ((d) . py_get"),
        "double-wrapped dict.get store, generated: {}",
        out
    );
    assert!(out.contains("alias = result"), "generated: {}", out);
    assert!(
        !out.contains("Some (result)"),
        "double-wrapped optional-name store, generated: {}",
        out
    );
}

#[test]
fn conditional_stores_into_optional_names_wrap_per_arm() {
    // `x if c else None` into a None-seeded name wraps each arm
    // independently: Some(x) / None. Wrapping the whole conditional would
    // bury the None arm as Some(None) and flip a later `is None` check.
    let src = concat!(
        "def f(n: int) -> int:\n",
        "    tag = None\n",
        "    tag = n if n > 0 else None\n",
        "    if tag is None:\n",
        "        return 0\n",
        "    return 1\n",
    );
    let out = compile(src, "optifexp.py");
    assert!(
        out.contains("tag = if") && out.contains("Some (n)"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("Some (if"),
        "wrapped the whole conditional, generated: {}",
        out
    );
}

#[test]
fn conditional_with_option_arms_stores_without_rewrap() {
    // Both arms already yield an Option (dict.get / None): the conditional
    // is an Option and stores through unchanged.
    let src = concat!(
        "def f(d: dict[int, int], n: int) -> int:\n",
        "    choice = None\n",
        "    choice = d.get(n) if n > 0 else None\n",
        "    if choice is None:\n",
        "        return -1\n",
        "    return 0\n",
    );
    let out = compile(src, "optifexp2.py");
    assert!(
        out.contains("choice = if"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("Some (if") && !out.contains("Some ((d) . py_get"),
        "double-wrapped a conditional Option, generated: {}",
        out
    );
}

#[test]
fn conditional_arguments_to_optional_parameters_wrap_per_arm() {
    let src = concat!(
        "def label(tag: Optional[int]) -> int:\n",
        "    return 0\n",
        "\n",
        "def f(n: int) -> int:\n",
        "    return label(n if n > 0 else None)\n",
    );
    let out = compile(src, "optifexp3.py");
    assert!(
        out.contains("label (if") && out.contains("Some (n)"),
        "generated: {}",
        out
    );
    assert!(
        !out.contains("Some (if"),
        "wrapped the whole conditional argument, generated: {}",
        out
    );
}

#[test]
fn optional_returning_calls_store_and_pass_without_rewrap() {
    // find() generates Result<Option<i64>, PyException>; the call site's `?`
    // leaves an Option, which must flow into optional names and Optional
    // parameters as-is.
    let src = concat!(
        "def find(d: dict[str, int], k: str) -> Optional[int]:\n",
        "    return d.get(k)\n",
        "\n",
        "def label(tag: Optional[int]) -> int:\n",
        "    return 0\n",
        "\n",
        "def f(d: dict[str, int]) -> int:\n",
        "    hit = None\n",
        "    hit = find(d, \"a\")\n",
        "    return label(find(d, \"b\"))\n",
    );
    let out = compile(src, "optret.py");
    assert!(out.contains("hit = find"), "generated: {}", out);
    assert!(
        !out.contains("hit = Some (find"),
        "double-wrapped Optional-returning call store, generated: {}",
        out
    );
    assert!(
        !out.contains("label (Some (find"),
        "double-wrapped Optional-returning call argument, generated: {}",
        out
    );
}

#[test]
fn typing_imports_lower_to_nothing() {
    let out = compile("from typing import Optional\nx = 1\n", "typing.py");
    assert!(!out.contains("typing"), "generated: {}", out);
}

#[test]
fn membership_uses_py_contains() {
    let out = compile("found = x in items", "in.py");
    assert!(out.contains("py_contains"), "generated: {}", out);

    let out = compile("missing = x not in items", "notin.py");
    assert!(out.contains("! (items) . py_contains"), "generated: {}", out);
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

// ---- Struct-based classes ----

fn compile_err(src: &str, name: &str) -> String {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed: {}", e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let err = module
        .to_rust(
            CodeGenContext::Module(name.replace(".py", "")),
            PythonOptions::default(),
            symbols,
        )
        .expect_err("conversion must fail loudly");
    format!("{}", err)
}

const COUNTER: &str = concat!(
    "class Counter:\n",
    "    def __init__(self, label: str, start: int = 0):\n",
    "        self.label = label\n",
    "        self.count = start\n",
    "\n",
    "    def bump(self, amount: int) -> int:\n",
    "        self.count += amount\n",
    "        return self.count\n",
    "\n",
    "    def double_bump(self, amount: int) -> int:\n",
    "        self.bump(amount)\n",
    "        self.bump(amount)\n",
    "        return self.count\n",
    "\n",
    "    def peek(self) -> int:\n",
    "        return self.count\n",
);

#[test]
fn classes_lower_to_structs_with_inferred_fields() {
    let out = compile(COUNTER, "counter.py");
    assert!(out.contains("pub struct Counter"), "generated: {}", out);
    assert!(out.contains("pub label : String"), "generated: {}", out);
    assert!(out.contains("pub count : i64"), "generated: {}", out);
    assert!(
        out.contains("pub fn new (label : impl Into < String > , start : i64) -> Result < Self , PyException >"),
        "generated: {}",
        out
    );
    assert!(
        out.contains("__rython_self . __init__ (label , start) ?"),
        "generated: {}",
        out
    );
}

#[test]
fn method_receivers_follow_mutation_including_transitive_calls() {
    let out = compile(COUNTER, "receivers.py");
    // __init__ and bump store through self; double_bump only via calling
    // bump; peek reads only.
    assert!(out.contains("fn __init__ (& mut self ,"), "generated: {}", out);
    assert!(out.contains("fn bump (& mut self ,"), "generated: {}", out);
    assert!(
        out.contains("fn double_bump (& mut self ,"),
        "transitive self-call must select &mut self: {}",
        out
    );
    assert!(out.contains("fn peek (& self ,"), "generated: {}", out);
}

#[test]
fn construction_and_method_calls_propagate_exceptions() {
    let src = format!(
        "{}\n\ndef run() -> int:\n    c = Counter(\"hits\")\n    c.bump(amount=2)\n    return c.peek()\n",
        COUNTER
    );
    let out = compile(&src, "classcalls.py");
    // Construction resolves defaults against __init__ (minus self) and
    // lowers to new()?; the omitted `start` fills with its default.
    assert!(
        out.contains("Counter :: new (\"hits\" , 0) ?"),
        "generated: {}",
        out
    );
    // Keyword arguments map against the method signature; calls take `?`.
    assert!(out.contains("(c) . bump (2) ?"), "generated: {}", out);
    assert!(out.contains("(c) . peek () ?"), "generated: {}", out);
    // A local constructing a mutating class needs a mutable binding.
    assert!(out.contains("let mut c ;"), "generated: {}", out);
}

#[test]
fn user_methods_shadow_builtin_method_rewrites() {
    // A user-defined method named like a dict/list builtin must resolve to
    // the class, not the py_get rewrite.
    let src = concat!(
        "class Box:\n",
        "    def __init__(self, v: int):\n",
        "        self.v = v\n",
        "\n",
        "    def get(self, bonus: int) -> int:\n",
        "        return self.v + bonus\n",
        "\n",
        "def run() -> int:\n",
        "    b = Box(3)\n",
        "    return b.get(1)\n",
    );
    let out = compile(src, "shadow.py");
    assert!(out.contains("(b) . get (1) ?"), "generated: {}", out);
    assert!(!out.contains("py_get"), "generated: {}", out);
}

#[test]
fn composed_fields_type_and_resolve_through_chains() {
    let src = concat!(
        "class Point:\n",
        "    def __init__(self, x: int):\n",
        "        self.x = x\n",
        "\n",
        "    def shift(self, dx: int):\n",
        "        self.x += dx\n",
        "\n",
        "class Holder:\n",
        "    def __init__(self, p: Point):\n",
        "        self.p = p\n",
        "\n",
        "    def nudge(self):\n",
        "        self.p.shift(1)\n",
    );
    let out = compile(src, "compose.py");
    assert!(out.contains("pub p : Point"), "generated: {}", out);
    // shift mutates Point, so nudge mutates self through the field chain.
    assert!(out.contains("fn nudge (& mut self ,"), "generated: {}", out);
    assert!(
        out.contains(". shift (1) ?"),
        "field-chain method calls propagate exceptions: {}",
        out
    );
}

#[test]
fn unsupported_class_constructs_error_loudly() {
    let err = compile_err(
        "class Base:\n    pass\n\nclass Child(Base):\n    pass\n",
        "inherit.py",
    );
    assert!(err.contains("inheritance"), "error: {}", err);

    let err = compile_err("class C:\n    VERSION = 3\n", "classattr.py");
    assert!(err.contains("class attribute"), "error: {}", err);

    let err = compile_err(
        "class C:\n    def __init__(self):\n        self.x = None\n",
        "noneattr.py",
    );
    assert!(err.contains("cannot infer a type"), "error: {}", err);
}

#[test]
fn str_getters_clone_the_field_out_of_the_shared_receiver() {
    // `def name(self) -> str: return self.name` reads a String field
    // through &self: the return clones it — semantically exact, since
    // Python strings are immutable.
    let src = concat!(
        "class Tag:\n",
        "    def __init__(self, name: str):\n",
        "        self.name = name\n",
        "\n",
        "    def get_name(self) -> str:\n",
        "        return self.name\n",
    );
    let out = compile(src, "getter.py");
    assert!(
        out.contains("Ok ((self . name) . clone ())"),
        "generated: {}",
        out
    );
}

#[test]
fn class_method_named_new_errors_loudly() {
    let err = compile_err(
        "class C:\n    def new(self) -> int:\n        return 1\n",
        "newclash.py",
    );
    assert!(err.contains("`new`"), "error: {}", err);
    assert!(err.contains("constructor"), "error: {}", err);
}

#[test]
fn read_only_methods_with_mutator_names_do_not_force_mut() {
    // A user method shadowing a builtin mutator name (`pop`) that only
    // reads must not force a mutable receiver binding — class resolution
    // is authoritative over the syntactic method-name list.
    let src = concat!(
        "class Box:\n",
        "    def __init__(self, v: int):\n",
        "        self.v = v\n",
        "\n",
        "    def pop(self) -> int:\n",
        "        return self.v\n",
        "\n",
        "def run() -> int:\n",
        "    b = Box(3)\n",
        "    return b.pop()\n",
    );
    let out = compile(src, "romut.py");
    assert!(out.contains("fn pop (& self ,"), "generated: {}", out);
    assert!(
        out.contains("let b ;") && !out.contains("let mut b ;"),
        "read-only pop must not force `mut`: {}",
        out
    );
}

#[test]
fn mutations_inside_keyword_arguments_are_detected() {
    // `use_it(n=c.bump(2))` mutates `c` through a keyword-argument value;
    // the binding must be mutable.
    let src = concat!(
        "class Counter:\n",
        "    def __init__(self, start: int):\n",
        "        self.count = start\n",
        "\n",
        "    def bump(self, amount: int) -> int:\n",
        "        self.count += amount\n",
        "        return self.count\n",
        "\n",
        "def use_it(n: int) -> int:\n",
        "    return n\n",
        "\n",
        "def run() -> int:\n",
        "    c = Counter(1)\n",
        "    return use_it(n=c.bump(2))\n",
    );
    let out = compile(src, "kwmut.py");
    assert!(
        out.contains("let mut c ;"),
        "keyword-nested mutation must mark `c` mutable: {}",
        out
    );
}

#[test]
fn split_keyword_arguments_map_or_error_loudly() {
    // maxsplit by keyword maps to the right runtime variant...
    let out = compile(
        "def f(s: str):\n    return s.split(\",\", maxsplit=1)\n",
        "kwsplit.py",
    );
    assert!(
        out.contains("py_split_maxsplit (& (\",\") , 1) ?"),
        "generated: {}",
        out
    );
    // ...including whitespace mode with a keyword-only maxsplit.
    let out = compile(
        "def f(s: str):\n    return s.rsplit(maxsplit=2)\n",
        "kwrsplit.py",
    );
    assert!(
        out.contains("py_rsplit_whitespace_maxsplit (2)"),
        "generated: {}",
        out
    );
    // Unknown keywords are loud conversion errors, not silent drops.
    let err = compile_err(
        "def f(s: str):\n    return s.split(\",\", bogus=1)\n",
        "kwbad.py",
    );
    assert!(err.contains("unexpected keyword"), "error: {}", err);
    // Keywords on positional-only builtin methods fall through to the
    // loud no-signature error instead of being dropped.
    let err = compile_err(
        "def f(s: str):\n    return s.ljust(5, fillchar=\".\")\n",
        "kwljust.py",
    );
    assert!(err.contains("signature"), "error: {}", err);
}

// ---- str.format ----

#[test]
fn str_format_lowers_to_format_macro() {
    let out = compile(
        "def f(a: int, b: str) -> str:\n    return \"{} and {}\".format(a, b)\n",
        "fmt1.py",
    );
    assert!(out.contains("format !"), "generated: {}", out);
    assert!(out.contains("__rython_fmt0"), "generated: {}", out);

    // Positional reuse, keywords, and specs translate.
    let out = compile(
        "def f(x: float) -> str:\n    return \"{0} {0} {v:.2f}\".format(x, v=x)\n",
        "fmt2.py",
    );
    assert!(out.contains("__rython_fmt_v"), "generated: {}", out);
}

#[test]
fn str_format_errors_are_loud() {
    // Mixing auto and manual numbering is Python's ValueError.
    let err = compile_err(
        "def f(a: int, b: int) -> str:\n    return \"{} {1}\".format(a, b)\n",
        "fmtmix.py",
    );
    assert!(err.contains("automatic field numbering"), "error: {}", err);

    // A template name with no matching keyword.
    let err = compile_err(
        "def f() -> str:\n    return \"{missing}\".format(present=1)\n",
        "fmtname.py",
    );
    assert!(err.contains("missing"), "error: {}", err);

    // Specs Rust renders differently are rejected, not approximated.
    let err = compile_err(
        "def f(x: int) -> str:\n    return \"{:,}\".format(x)\n",
        "fmtgroup.py",
    );
    assert!(err.contains("thousands separator"), "error: {}", err);

    // Non-literal templates can't be checked at conversion time.
    let err = compile_err(
        "def f(t: str, x: int) -> str:\n    return t.format(x)\n",
        "fmtdyn.py",
    );
    assert!(err.contains("non-literal template"), "error: {}", err);
}

#[test]
fn fstring_specs_translate_or_error_loudly() {
    let out = compile(
        "def f(n: int) -> str:\n    return f\"{n:05d}|{n:>4}\"\n",
        "fspec.py",
    );
    assert!(out.contains("{:05}"), "generated: {}", out);
    assert!(out.contains("{:>4}"), "generated: {}", out);

    // The old behavior silently fell back to {} for unsupported specs;
    // now they fail at conversion time.
    let err = compile_err(
        "def f(x: float) -> str:\n    return f\"{x:e}\"\n",
        "fspecbad.py",
    );
    assert!(err.contains("presentation type"), "error: {}", err);
}

#[test]
fn repr_conversion_keeps_its_format_spec() {
    // "{0!r:>10}" pads the repr — the spec must not be dropped.
    let out = compile(
        "def f(n: int) -> str:\n    return \"{0!r:>10}\".format(n)\n",
        "reprspec.py",
    );
    assert!(out.contains(":>10?}"), "generated: {}", out);

    let out = compile(
        "def f(n: int) -> str:\n    return f\"{n!r:>10}\"\n",
        "freprspec.py",
    );
    assert!(out.contains(":>10?}"), "generated: {}", out);

    // Numeric presentation types on a repr are Python errors; loud here.
    let err = compile_err(
        "def f(n: int) -> str:\n    return \"{0!r:.2f}\".format(n)\n",
        "reprbad.py",
    );
    assert!(err.contains("cannot combine"), "error: {}", err);
}

#[test]
fn bare_precision_without_type_errors_loudly() {
    // Python's "{:.3}" on a float is GENERAL format (significant figures,
    // possibly scientific); Rust's is fixed decimals. Unknowable operand
    // type means loud rejection, pointing at .Ns / .Nf.
    let err = compile_err(
        "def f(x: float) -> str:\n    return \"{:.3}\".format(x)\n",
        "barep.py",
    );
    assert!(err.contains("presentation type is ambiguous"), "error: {}", err);
    let err = compile_err(
        "def f(x: float) -> str:\n    return f\"{x:.3}\"\n",
        "barepf.py",
    );
    assert!(err.contains("presentation type is ambiguous"), "error: {}", err);
}

// ---- Module-level globals and entry points ----

#[test]
fn module_constants_lower_to_statics() {
    let out = compile(
        concat!(
            "PI = 3.14159\n",
            "GREETING = \"hello\"\n",
            "DEBUG = True\n",
            "OFFSET = -3\n",
            "\n",
            "def area(r: float) -> float:\n",
            "    return PI * r * r\n",
        ),
        "consts.py",
    );
    assert!(out.contains("pub static PI : f64 = 3.14159"), "generated: {}", out);
    assert!(
        out.contains("pub static GREETING : & 'static str = \"hello\""),
        "generated: {}",
        out
    );
    assert!(out.contains("pub static DEBUG : bool = true"), "generated: {}", out);
    assert!(out.contains("pub static OFFSET : i64 = - 3"), "generated: {}", out);

    // A reassigned module name is NOT a constant; it keeps the old
    // module-init lowering.
    let out = compile("X = 1\nX = 2\n", "reassigned.py");
    assert!(!out.contains("pub static X"), "generated: {}", out);
}

#[test]
fn value_returning_main_gets_a_wrapper_entry_point() {
    // `def main() -> int` cannot be the Rust entry point (Result<i64, _>
    // does not implement Termination); the wrapper discards the value like
    // Python's `if __name__: main()` does.
    let out = compile(
        concat!(
            "def main() -> int:\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
        "intmain.py",
    );
    assert!(out.contains("fn python_main ()"), "generated: {}", out);
    assert!(
        out.contains("fn main () {"),
        "wrapper entry point expected: {}",
        out
    );
}

#[test]
fn integral_float_literals_keep_their_float_type() {
    // 2.0 must stay a float literal: Rust's Display drops the ".0" and the
    // re-parse would silently produce an integer (2.0 / 4 is 0.5 in
    // Python, but 2 / 4 as integers is 0).
    let out = compile("def f() -> float:\n    y = 2.0\n    return y\n", "flit.py");
    assert!(out.contains("y = 2.0"), "generated: {}", out);
    assert!(!out.contains("y = 2 ;"), "generated: {}", out);
}

#[test]
fn conditionally_reassigned_module_names_are_not_constants() {
    // DEBUG = False overwritten inside a module-level `if` must NOT freeze
    // as a static: the nested store would land on a shadowing local inside
    // __module_init__ while functions read the stale static.
    let out = compile(
        "DEBUG = False\nif 1 > 0:\n    DEBUG = True\n",
        "condglobal.py",
    );
    assert!(!out.contains("pub static DEBUG"), "generated: {}", out);

    // A for-loop target at module level is rebound each iteration.
    let out = compile("I = 0\nfor I in [1, 2]:\n    pass\n", "forglobal.py");
    assert!(!out.contains("pub static I"), "generated: {}", out);

    // Reassignment inside a module-level try body.
    let out = compile(
        "MODE = \"a\"\ntry:\n    MODE = \"b\"\nexcept ValueError:\n    pass\n",
        "tryglobal.py",
    );
    assert!(!out.contains("pub static MODE"), "generated: {}", out);
}

// ---------------------------------------------------------------------------
// no_std profile: OS-facing constructs fail at conversion time
// ---------------------------------------------------------------------------

fn compile_nostd(src: &str, name: &str) -> Result<String, String> {
    let module = parse(src, name).unwrap_or_else(|e| panic!("parse failed: {}", e));
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let options = PythonOptions {
        no_std: true,
        ..Default::default()
    };
    module
        .to_rust(CodeGenContext::Module(name.replace(".py", "")), options, symbols)
        .map(|tokens| tokens.to_string())
        .map_err(|e| python_ast::format_error_chain(e.as_ref()))
}

#[test]
fn nostd_modules_carry_an_alloc_prelude() {
    // Under #![no_std] the prelude has no String/Vec/format!; every module
    // brings the alloc surface generated code leans on into scope itself.
    let out = compile_nostd("def f(n: int) -> str:\n    return f\"n={n}\"\n", "np.py")
        .expect("OS-free module must convert");
    assert!(out.contains("extern crate alloc"), "generated: {}", out);
    assert!(out.contains("use alloc ::"), "generated: {}", out);

    // The std profile stays exactly as before: no alloc plumbing.
    let std_out = compile("def f(n: int) -> str:\n    return f\"n={n}\"\n", "sp.py");
    assert!(!std_out.contains("extern crate alloc"), "generated: {}", std_out);
}

#[test]
fn nostd_io_builtins_error_loudly() {
    for src in ["print(\"hi\")\n", "x = input()\n", "f = open(\"a.txt\")\n"] {
        let err = compile_nostd(src, "io.py").expect_err("I/O builtin must fail");
        assert!(err.contains("no_std profile"), "{:?}: {}", src, err);
    }

    // A user definition shadows the builtin as usual and stays convertible.
    let out = compile_nostd(
        "def print(s: str) -> str:\n    return s\n\ndef f() -> str:\n    return print(\"x\")\n",
        "shadow.py",
    )
    .expect("shadowed print is the user's own function");
    assert!(out.contains("fn print"), "generated: {}", out);
}

#[test]
fn nostd_std_tier_imports_error_loudly() {
    for src in [
        "import os\n",
        "import sys\n",
        "from datetime import datetime\n",
        "import math\n",
        "from os.path import join\n",
    ] {
        let err = compile_nostd(src, "imp.py").expect_err("std-tier import must fail");
        assert!(err.contains("std tier"), "{:?}: {}", src, err);
    }

    // alloc-tier runtime modules stay importable.
    for src in ["import json\n", "import collections\n", "import itertools\n"] {
        compile_nostd(src, "ok.py").unwrap_or_else(|e| {
            panic!("alloc-tier import must convert: {:?}: {}", src, e)
        });
    }
}

#[test]
fn nostd_main_blocks_error_loudly() {
    let err = compile_nostd(
        "def main() -> int:\n    return 0\n\nif __name__ == \"__main__\":\n    main()\n",
        "entry.py",
    )
    .expect_err("__main__ needs a process entry point");
    assert!(err.contains("no_std profile"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// Builtin lowering: min/max/sorted/enumerate/pow/len/repr/reversed
// ---------------------------------------------------------------------------

#[test]
fn min_max_lower_to_variant_functions_with_exception_propagation() {
    // Single-iterable form raises on empty, so it propagates with `?`.
    let out = compile("def f(xs: list[int]) -> int:\n    return min(xs)\n", "m1.py");
    assert!(out.contains("min (& (xs)) ?"), "generated: {}", out);

    // Two and three scalar arguments fold pairwise.
    let out = compile("def f(a: int, b: int) -> int:\n    return max(a, b)\n", "m2.py");
    assert!(out.contains("max2 (a , b)"), "generated: {}", out);
    let out = compile(
        "def f(a: int, b: int, c: int) -> int:\n    return min(a, b, c)\n",
        "m3.py",
    );
    assert!(out.contains("min2 (min2 (a , b) , c)"), "generated: {}", out);

    // default= never raises; key= does.
    let out = compile(
        "def f(xs: list[int]) -> int:\n    return min(xs, default=7)\n",
        "m4.py",
    );
    assert!(out.contains("min_default (& (xs) , 7)"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> int:\n    return max(xs, key=lambda x: -x)\n",
        "m5.py",
    );
    assert!(out.contains("max_key (& (xs) ,"), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);

    // Unknown keywords stay loud.
    let err = compile_err("x = min([1], foo=2)\n", "m6.py");
    assert!(err.contains("unexpected"), "error: {}", err);
}

#[test]
fn sorted_lowers_by_keyword_combination() {
    let out = compile("def f(xs: list[int]) -> list[int]:\n    return sorted(xs)\n", "s1.py");
    assert!(out.contains("sorted (& (xs))"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs, reverse=True)\n",
        "s2.py",
    );
    assert!(out.contains("sorted_reverse (& (xs) , true)"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs, key=lambda x: -x)\n",
        "s3.py",
    );
    assert!(out.contains("sorted_key (& (xs) ,"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return sorted(xs, key=lambda x: -x, reverse=True)\n",
        "s4.py",
    );
    assert!(out.contains("sorted_key_reverse (& (xs) ,"), "generated: {}", out);
}

#[test]
fn enumerate_start_and_pow_arities_lower_to_their_variants() {
    let out = compile(
        "for i, x in enumerate([10, 20], start=5):\n    pass\n",
        "e1.py",
    );
    assert!(out.contains("enumerate_start ("), "generated: {}", out);
    let out = compile("for i, x in enumerate([10]):\n    pass\n", "e2.py");
    assert!(out.contains("enumerate ("), "generated: {}", out);
    assert!(!out.contains("enumerate_start"), "generated: {}", out);

    let out = compile("y = pow(2, 5)\n", "p1.py");
    assert!(out.contains("pow (2 , 5)"), "generated: {}", out);
    let out = compile("y = pow(2, 5, 7)\n", "p2.py");
    assert!(out.contains("pow_mod (2 , 5 , 7) ?"), "generated: {}", out);
}

#[test]
fn by_reference_builtins_borrow_their_argument() {
    // len/repr/reversed take references at the runtime layer; Python's
    // calls never consume the value.
    let out = compile("def f(xs: list[int]) -> int:\n    return len(xs)\n", "b1.py");
    assert!(out.contains("len (& (xs))"), "generated: {}", out);
    let out = compile("def f(xs: list[int]) -> str:\n    return repr(xs)\n", "b2.py");
    assert!(out.contains("repr (& (xs))"), "generated: {}", out);
    let out = compile(
        "def f(xs: list[int]) -> list[int]:\n    return reversed(xs)\n",
        "b3.py",
    );
    assert!(out.contains("reversed (& (xs))"), "generated: {}", out);

    // A user-defined function of the same name shadows the builtin shape.
    let out = compile(
        "def len(x: int) -> int:\n    return x\n\ndef g(v: int) -> int:\n    return len(v)\n",
        "b4.py",
    );
    assert!(out.contains("len (v)"), "generated: {}", out);
}

// ---------------------------------------------------------------------------
// datetime constructors, strptime, and runtime-module imports
// ---------------------------------------------------------------------------

#[test]
fn datetime_constructors_map_keywords_onto_new() {
    let out = compile(
        "from datetime import timedelta\ntd = timedelta(days=1, hours=2)\n",
        "td.py",
    );
    assert!(
        out.contains("timedelta :: new (Some (1) , None , None , None , None , Some (2) , None)"),
        "generated: {}",
        out
    );
    let out = compile(
        "from datetime import date\nd = date(2024, 3, 1)\n",
        "d.py",
    );
    assert!(out.contains("date :: new (2024 , 3 , 1) ?"), "generated: {}", out);
    let out = compile(
        "from datetime import datetime\ndt = datetime(2024, 3, 1, hour=10)\n",
        "dt.py",
    );
    assert!(
        out.contains("datetime :: new (2024 , 3 , 1 , Some (10) , None , None , None) ?"),
        "generated: {}",
        out
    );

    // Unknown keywords and missing required arguments stay loud.
    let err = compile_err(
        "from datetime import timedelta\ntd = timedelta(fortnights=1)\n",
        "tde.py",
    );
    assert!(err.contains("unexpected keyword"), "error: {}", err);
    let err = compile_err("from datetime import date\nd = date(2024)\n", "de.py");
    assert!(err.contains("missing required argument"), "error: {}", err);
}

#[test]
fn strptime_and_module_attribute_calls_lower_to_paths() {
    let out = compile(
        "from datetime import datetime\ndt = datetime.strptime(\"x\", \"%Y\")\n",
        "sp.py",
    );
    assert!(
        out.contains("datetime :: strptime (\"x\" , \"%Y\") ?"),
        "generated: {}",
        out
    );
    let out = compile("import time\nt = time.monotonic()\n", "tm.py");
    assert!(out.contains("time :: monotonic ()"), "generated: {}", out);
}

#[test]
fn runtime_module_imports_lower_to_nothing_and_aliases_stay_loud() {
    // The modules are already in scope via `use stdpython::*`; a bare
    // `use math;` would not even resolve.
    let out = compile("import math\nimport random\n", "imp.py");
    assert!(!out.contains("use math"), "generated: {}", out);
    assert!(!out.contains("use random"), "generated: {}", out);

    let err = compile_err("import time as t\n", "alias.py");
    assert!(err.contains("aliasing"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// itertools lowering: keyword variants and by-reference iterables
// ---------------------------------------------------------------------------

#[test]
fn itertools_keyword_spellings_lower_to_variants() {
    let base = "from itertools import accumulate, product, zip_longest, groupby\n";
    let out = compile(&format!("{}a = accumulate([1, 2])\n", base), "i1.py");
    assert!(out.contains("accumulate_sum (& (vec ! [1 , 2]))"), "generated: {}", out);
    let out = compile(
        &format!("{}a = accumulate([1, 2], initial=10)\n", base),
        "i2.py",
    );
    assert!(out.contains("accumulate_sum_initial ("), "generated: {}", out);
    let out = compile(
        &format!("{}a = accumulate([1, 2], lambda x, y: x * y)\n", base),
        "i3.py",
    );
    assert!(out.contains("accumulate_func ("), "generated: {}", out);

    let out = compile(&format!("{}p = product([1], [2])\n", base), "i4.py");
    assert!(out.contains("product2 ("), "generated: {}", out);
    let out = compile(&format!("{}p = product([1], repeat=2)\n", base), "i5.py");
    assert!(out.contains("product_repeat2 ("), "generated: {}", out);
    // repeat must be a literal arity — tuple width is a compile-time shape.
    let err = compile_err(&format!("{}p = product([1], repeat=5)\n", base), "i6.py");
    assert!(err.contains("literal 2 or 3"), "error: {}", err);

    let out = compile(
        &format!("{}z = zip_longest([1], [2], fillvalue=0)\n", base),
        "i7.py",
    );
    assert!(out.contains("zip_longest_fill ("), "generated: {}", out);
    let out = compile(
        &format!("{}g = groupby([1], key=lambda x: x)\n", base),
        "i8.py",
    );
    assert!(out.contains("groupby_key ("), "generated: {}", out);

    // Unknown keywords stay loud.
    let err = compile_err(&format!("{}g = groupby([1], foo=1)\n", base), "i9.py");
    assert!(err.contains("unexpected"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// functools/heapq/copy/textwrap lowering, and mutating methods on
// subscripted receivers
// ---------------------------------------------------------------------------

#[test]
fn pure_module_calls_lower_with_borrows_and_arity_variants() {
    let out = compile(
        "from functools import reduce\nr = reduce(lambda a, b: a + b, [1, 2])\n",
        "f1.py",
    );
    assert!(out.contains("reduce ("), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);
    let out = compile(
        "from functools import reduce\nr = reduce(lambda a, b: a + b, [1, 2], 10)\n",
        "f2.py",
    );
    assert!(out.contains("reduce_initial ("), "generated: {}", out);

    // heapq mutates its first argument: &mut lowering and a mut binding.
    let out = compile(
        "from heapq import heappush, heappop\nh = [3, 1]\nheappush(h, 2)\nx = heappop(h)\n",
        "h1.py",
    );
    assert!(out.contains("heappush (& mut (h) , 2)"), "generated: {}", out);
    assert!(out.contains("heappop (& mut (h)) ?"), "generated: {}", out);
    assert!(out.contains("let mut h"), "heap binding must be mut: {}", out);

    // Module-attribute spelling lowers to the same shapes AND marks the
    // heap binding mutable (Devin review on #53: only the bare-function
    // spelling used to).
    let out = compile("import heapq\nh = [2, 1]\nheapq.heapify(h)\n", "h2.py");
    assert!(out.contains("heapq :: heapify (& mut (h))"), "generated: {}", out);
    assert!(out.contains("let mut h"), "heap binding must be mut: {}", out);

    let out = compile("from copy import deepcopy\nc = deepcopy([1])\n", "c1.py");
    assert!(out.contains("deepcopy (& ("), "generated: {}", out);
    let out = compile(
        "from textwrap import indent\ns = indent(\"a\", \"> \")\n",
        "t1.py",
    );
    assert!(out.contains("indent (& (\"a\") , & (\"> \"))"), "generated: {}", out);
}

#[test]
fn mutating_methods_on_subscripted_receivers_use_the_place_lowering() {
    // xs[0].append(v) must mutate the real element: the Load lowering
    // (py_index) yields a clone and the write would silently vanish.
    let out = compile("xs = [[1], [2]]\nxs[0].append(9)\n", "sub1.py");
    assert!(
        out.contains("py_index_mut (0) ?) . push (9)"),
        "generated: {}",
        out
    );
    // Read-only methods keep the Load lowering.
    let out = compile("xs = [[1]]\nn = xs[0].count(1)\n", "sub2.py");
    assert!(!out.contains("py_index_mut"), "generated: {}", out);

    // The heapq mutators' heap argument is a place too: heappush(rows[i], v)
    // through the Load path would push into a clone.
    let out = compile(
        "from heapq import heappush\nrows = [[1], [2]]\nheappush(rows[0], 5)\n",
        "sub3.py",
    );
    assert!(
        out.contains("heappush ((rows) . py_index_mut (0) ? , 5)"),
        "generated: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// re module lowering
// ---------------------------------------------------------------------------

#[test]
fn re_calls_lower_to_borrowing_fallible_paths() {
    let out = compile("import re\nm = re.search(r\"\\d\", \"a1\")\n", "r1.py");
    assert!(
        out.contains("re :: search (& (\"\\\\d\") , & (\"a1\") , \"\") ?"),
        "generated: {}",
        out
    );
    // `match` is a Rust keyword: the runtime function is r#match.
    let out = compile("import re\nm = re.match(r\"\\d\", \"1\")\n", "r2.py");
    assert!(out.contains("re :: r#match ("), "generated: {}", out);
    let out = compile(
        "import re\ns = re.sub(r\"a\", \"b\", \"aa\")\n",
        "r3.py",
    );
    assert!(out.contains("re :: sub ("), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);
    // m.group() lowers to group(0).
    let out = compile(
        "import re\nm = re.search(r\"a\", \"a\")\ng = m.group()\n",
        "r4.py",
    );
    assert!(out.contains(". group (0)"), "generated: {}", out);
    // from-import spelling, including the keyword-name function.
    let out = compile(
        "from re import findall, match\nxs = findall(r\"a\", \"aa\")\nm = match(r\"a\", \"ab\")\n",
        "r5.py",
    );
    assert!(out.contains("findall (& ("), "generated: {}", out);
    assert!(out.contains("r#match (& ("), "generated: {}", out);
    // Flags lower to inline flag letters; unknown flags are loud.
    let out = compile(
        "import re\nxs = re.findall(r\"a\", \"A\", re.IGNORECASE)\n",
        "r6.py",
    );
    assert!(out.contains("\"i\") ?"), "generated: {}", out);
    let out = compile(
        "import re\nxs = re.findall(r\"a\", \"A\", flags=re.IGNORECASE | re.MULTILINE)\n",
        "r7.py",
    );
    assert!(out.contains("\"im\") ?"), "generated: {}", out);
    let out = compile(
        "import re\ns = re.sub(r\"a\", \"b\", \"aa\", count=1)\n",
        "r8.py",
    );
    assert!(out.contains(", 1 , \"\") ?"), "generated: {}", out);
    let err = compile_err(
        "import re\nxs = re.findall(r\"a\", \"A\", re.VERBOSE)\n",
        "r9.py",
    );
    assert!(err.contains("unsupported re flag"), "error: {}", err);
    // split's THIRD positional is maxsplit (not flags, unlike the rest).
    let out = compile(
        "import re\nxs = re.split(r\"a\", \"b\", 1)\n",
        "r10.py",
    );
    assert!(out.contains("re :: split (& (\"a\") , & (\"b\") , 1 , \"\") ?"), "generated: {}", out);
    let out = compile(
        "import re\nxs = re.split(r\"a\", \"b\", maxsplit=2, flags=re.IGNORECASE)\n",
        "r11.py",
    );
    assert!(out.contains(", 2 , \"i\") ?"), "generated: {}", out);
    // Surplus positionals are loud, not silently dropped.
    let err = compile_err(
        "import re\nm = re.search(r\"a\", \"b\", re.IGNORECASE, 5)\n",
        "r12.py",
    );
    assert!(err.contains("at most 3"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// map/filter/list lowering
// ---------------------------------------------------------------------------

#[test]
fn map_filter_dispatch_on_the_function_arguments_shape() {
    // Lambdas are plain closures.
    let out = compile("ys = list(map(lambda x: x * 2, [1, 2]))\n", "mf1.py");
    assert!(out.contains("list (map (| x |"), "generated: {}", out);
    assert!(!out.contains("map_fallible"), "generated: {}", out);

    // User-defined functions return Result: the fallible variant + `?`.
    let out = compile(
        "def double(n: int) -> int:\n    return n * 2\n\nys = list(map(double, [1, 2]))\n",
        "mf2.py",
    );
    assert!(out.contains("map_fallible (double ,"), "generated: {}", out);
    assert!(out.contains(") ?"), "generated: {}", out);

    let out = compile("ys = filter(lambda x: x > 1, [1, 2, 3])\n", "mf3.py");
    assert!(out.contains("filter (| x |"), "generated: {}", out);
    // filter(None, xs) keeps truthy elements.
    let out = compile("ys = filter(None, [0, 1, 2])\n", "mf4.py");
    assert!(out.contains("filter_truthy ("), "generated: {}", out);

    // list() with no argument has no inferable type: loud.
    let err = compile_err("ys = list()\n", "mf5.py");
    assert!(err.contains("iterable argument"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// hashlib lowering and str.encode()
// ---------------------------------------------------------------------------

#[test]
fn hashlib_and_encode_lower_correctly() {
    let out = compile(
        "import hashlib\nh = hashlib.sha256(\"x\".encode())\n",
        "hl1.py",
    );
    assert!(
        out.contains("hashlib :: sha256 (& ((\"x\") . as_bytes () . to_vec ()))"),
        "generated: {}",
        out
    );
    // Zero-arg constructors map to the _new variants for the update idiom.
    let out = compile("from hashlib import sha256\nh = sha256()\n", "hl2.py");
    assert!(out.contains("sha256_new ()"), "generated: {}", out);
    // Only utf-8 encodings are supported — anything else is loud.
    let err = compile_err("s = \"x\".encode(\"latin-1\")\n", "hl3.py");
    assert!(err.contains("utf-8"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// textwrap.wrap/fill lowering
// ---------------------------------------------------------------------------

#[test]
fn wrap_and_fill_lower_with_width_defaults() {
    let out = compile("from textwrap import wrap\nxs = wrap(\"a b\")\n", "w1.py");
    assert!(out.contains("wrap (& (\"a b\") , 70) ?"), "generated: {}", out);
    let out = compile(
        "from textwrap import fill\ns = fill(\"a b\", width=9)\n",
        "w2.py",
    );
    assert!(out.contains("fill (& (\"a b\") , 9) ?"), "generated: {}", out);
    let out = compile(
        "import textwrap\nxs = textwrap.wrap(\"a b\", 12)\n",
        "w3.py",
    );
    assert!(out.contains("textwrap :: wrap (& (\"a b\") , 12) ?"), "generated: {}", out);
    // Unsupported options stay loud.
    let err = compile_err(
        "from textwrap import wrap\nxs = wrap(\"a\", initial_indent=\"> \")\n",
        "w4.py",
    );
    assert!(err.contains("unexpected keyword"), "error: {}", err);
}

// ---------------------------------------------------------------------------
// isinstance (static constant) and hash lowering
// ---------------------------------------------------------------------------

#[test]
fn isinstance_lowers_to_a_static_constant_or_a_loud_error() {
    // Annotated parameters decide at conversion time.
    let out = compile(
        "def f(n: int) -> bool:\n    return isinstance(n, int)\n",
        "is1.py",
    );
    assert!(out.contains("return Ok (true)") || out.contains("true"), "generated: {}", out);
    let out = compile(
        "def f(n: int) -> bool:\n    return isinstance(n, str)\n",
        "is2.py",
    );
    assert!(out.contains("false"), "generated: {}", out);
    // Literal-assigned locals count; bool is a subclass of int.
    let out = compile(
        "def f() -> bool:\n    x = 1.5\n    return isinstance(x, float)\n",
        "is3.py",
    );
    assert!(out.contains("true"), "generated: {}", out);
    let out = compile(
        "def f(b: bool) -> bool:\n    return isinstance(b, int)\n",
        "is4.py",
    );
    assert!(out.contains("true"), "generated: {}", out);
    let out = compile(
        "def f(n: int) -> bool:\n    return isinstance(n, bool)\n",
        "is5.py",
    );
    assert!(out.contains("false"), "generated: {}", out);

    // Unknown types are loud, not guessed.
    let err = compile_err(
        "def f(v):\n    return isinstance(v, int)\n",
        "is6.py",
    );
    assert!(err.contains("statically"), "error: {}", err);
}

#[test]
fn hash_lowers_by_reference() {
    let out = compile("h = hash(\"a\")\n", "hs1.py");
    assert!(out.contains("hash (& (\"a\"))"), "generated: {}", out);
}
