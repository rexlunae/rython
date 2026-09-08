//! Callables as VALUES (issue #122): a nested `def` and a `lambda`
//! lowered as `stdpython::PyCallable` closures, and the capture analysis
//! that keeps them faithful to Python.
//!
//! Python's closure is a set of CELLS, and it is LATE-BINDING: the
//! nested function does not copy the enclosing scope, it holds the cells
//! its free names are bound in and reads them when it is CALLED. So
//! `x = 1; f = lambda: x; x = 2; f()` is 2, a container the enclosing
//! scope mutates after the `def` is seen by the closure, and every
//! closure built in a loop shares the loop variable's one binding.
//!
//! rython's locals are values, so a captured name is held in a
//! `stdpython::PyCell` — the closure's `move` capture clones the cell,
//! which shares it, and both sides read and write ONE binding.
//!
//! The exception is a capture that CANNOT change after the definition:
//! bound exactly once, unconditionally, outside any loop, and mutated by
//! nobody. There is no later value for the closure to have missed, so the
//! clone and the cell cannot be told apart, and the clone is what is
//! emitted — which is every ordinary `make_adder(n)`-shaped closure.
//!
//! A LAMBDA captures through the same cells. What it cannot do is MUTATE
//! one: a cell's stores are decided per STATEMENT and a lambda is an
//! expression, so one that would mutate a capture is refused rather than
//! letting the mutation vanish — the nested `def` spelling carries it.
//!
//! What a closure cannot be is decided here too, once, and reported as a
//! refusal the caller turns into a loud error rather than a silently
//! dropped definition.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::tree::visit::{Descend, Flow, walk_stmts};
use crate::{
    CodeGenContext, ExprType, FunctionDef, PythonOptions, Statement, StatementType,
    SymbolTableScopes, TypeInfo,
};

/// What lowering a nested definition as a callable value requires.
#[derive(Clone, Debug, Default)]
pub struct ClosureInfo {
    /// Enclosing-scope names the closure reads, cloned into it at the
    /// definition point — the cells among them share, the rest copy.
    pub captures: Vec<String>,
    /// The subset held in a `stdpython::PyCell` shared with the enclosing
    /// scope: everything the closure or the enclosing scope can still
    /// change after the definition (see the module docs).
    pub cells: Vec<String>,
    /// Whether the enclosing scope HOISTS the definition's name. Python
    /// names are function-scoped, so a `def` under an `if` or in a `try`
    /// binds the function's name, not a block-local one: the closure
    /// stores into the hoisted binding exactly as an assignment does.
    pub hoisted: bool,
}

/// Why a nested `def` cannot be a callable value. `None` means it can.
///
/// Every refusal is a shape whose Rust type cannot be written down (an
/// unannotated parameter, a variadic arity) or whose Python semantics
/// the closure model does not carry (a generator, a rebinding
/// `nonlocal`). The caller reports the reason at the definition — a
/// dropped `def` that silently answers None is what this replaces.
pub(crate) fn closure_refusal(def: &FunctionDef) -> Option<String> {
    if !def.decorator_list.is_empty() {
        return Some(format!(
            "nested function `{}` is decorated; a decorator applies to a \
             definition, and a callable value has none",
            def.name
        ));
    }
    if def.args.vararg.is_some() || def.args.kwarg.is_some() {
        return Some(format!(
            "nested function `{}` takes `*args`/`**kwargs`; a callable value's \
             argument tuple has a fixed arity",
            def.name
        ));
    }
    if !def.args.kwonlyargs.is_empty() {
        return Some(format!(
            "nested function `{}` has a keyword-only parameter; a callable \
             value's argument tuple is positional and has no names to match \
             a keyword against",
            def.name
        ));
    }
    for p in def.args.posonlyargs.iter().chain(def.args.args.iter()) {
        if p.evaluated_annotation().is_none() {
            return Some(format!(
                "nested function `{}` has an unannotated parameter `{}`; a \
                 callable value's argument tuple needs every type",
                def.name, p.arg
            ));
        }
    }
    if !def.args.defaults.is_empty()
        || def.args.kw_defaults.iter().any(|d| d.is_some())
    {
        return Some(format!(
            "nested function `{}` has a defaulted parameter; Rust has no \
             default arguments and a callable value has no signature to \
             fill them in from",
            def.name
        ));
    }
    if def.returns.is_none() {
        return Some(format!(
            "nested function `{}` has no return annotation; a callable value's \
             type names what it returns",
            def.name
        ));
    }
    if crate::body_has_yields(&def.body) {
        return Some(format!(
            "nested function `{}` is a generator; a callable value returns \
             its result, not a suspended frame",
            def.name
        ));
    }
    if walk_stmts(&def.body, Descend::SkipDefs, &mut |s| {
        match &s.statement {
            StatementType::Nonlocal(_) | StatementType::Global(_) => Flow::Stop,
            _ => Flow::Continue,
        }
    }) {
        None
    } else {
        Some(format!(
            "nested function `{}` declares `nonlocal`/`global`; rebinding an \
             enclosing name from a callable value is not modeled (mutating \
             the object it is bound to is)",
            def.name
        ))
    }
}

/// The NAME a store target is rooted at, through subscripts and
/// attributes alike (`counter["n"]`, `self.rows[i].tag`). The call
/// module's `root_name` stops at a subscript — it answers a different
/// question (which receiver a call dispatches on); a store's root is the
/// object the write lands in.
fn store_root(expr: &ExprType) -> Option<&str> {
    match expr {
        ExprType::Name(n) => Some(&n.id),
        ExprType::Attribute(a) => store_root(&a.value),
        ExprType::Subscript(sub) => store_root(&sub.value),
        _ => None,
    }
}

/// The names a nested definition captures from `scope_names` — the
/// enclosing function's own bindings — and which of them it mutates.
///
/// A name the definition binds itself (a parameter, a local, its own
/// nested `def`) is not a capture; neither is a module-level function,
/// class or import, which the generated code names directly. What is
/// left is exactly Python's cell set.
pub(crate) fn closure_info(def: &FunctionDef, scope_names: &HashSet<String>) -> ClosureInfo {
    let mut captures: Vec<String> = Vec::new();
    let mut cells: HashSet<String> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();

    let consider = |name: &str, captures: &mut Vec<String>, seen: &mut HashSet<String>| {
        if !scope_names.contains(name) || crate::ast::tree::visit::def_owns_name(def, name) {
            return;
        }
        if seen.insert(name.to_string()) {
            captures.push(name.to_string());
        }
    };

    // Every name the body reads, its own nested definitions and lambda
    // bodies included: a doubly-nested closure's free names are captured
    // by this one first, exactly as Python's cells chain.
    walk_stmts(&def.body, Descend::All, &mut |s| {
        for e in crate::ast::tree::visit::stmt_all_exprs(s) {
            crate::ast::tree::visit::walk_expr(e, &mut |sub| {
                if let ExprType::Name(n) = sub {
                    consider(&n.id, &mut captures, &mut seen);
                }
            });
        }
        // A store THROUGH a captured name (`counter["n"] += 1`,
        // `self.seen.add(x)` is not one — the root must be the name
        // itself) makes it a cell: the enclosing scope must see the
        // change. A bare-name target rebinds the closure's OWN local
        // (Python needs `nonlocal` to reach the cell), so it is not one.
        for t in crate::ast::tree::visit::stmt_targets(s) {
            if matches!(t, ExprType::Name(_)) {
                continue;
            }
            if let Some(root) = store_root(t)
                && scope_names.contains(root)
                && !crate::ast::tree::visit::def_owns_name(def, root)
            {
                cells.insert(root.to_string());
            }
        }
        Flow::Continue
    });

    // A mutating METHOD on a captured name (`acc.append(x)`) mutates the
    // same object the enclosing scope holds.
    walk_stmts(&def.body, Descend::All, &mut |s| {
        for e in crate::ast::tree::visit::stmt_all_exprs(s) {
            crate::ast::tree::visit::walk_expr(e, &mut |sub| {
                if let ExprType::Call(c) = sub
                    && let ExprType::Attribute(attr) = c.func.as_ref()
                    && crate::ast::tree::scope::mutates_receiver(&attr.attr)
                    && let ExprType::Name(recv) = attr.value.as_ref()
                    && scope_names.contains(&recv.id)
                    && !crate::ast::tree::visit::def_owns_name(def, &recv.id)
                {
                    cells.insert(recv.id.clone());
                }
            });
        }
        Flow::Continue
    });

    let mut cells: Vec<String> = cells.into_iter().collect();
    cells.sort();
    // A cell is a capture too — the closure clones the shared handle.
    for c in &cells {
        if seen.insert(c.clone()) {
            captures.push(c.clone());
        }
    }
    ClosureInfo { captures, cells, hoisted: false }
}

/// The names a function scope binds: its parameters and everything its
/// body binds. The candidate set a nested definition captures FROM.
pub(crate) fn scope_binding_names(def_args: &crate::ParameterList, body: &[Statement]) -> HashSet<String> {
    let mut names: HashSet<String> = def_args
        .posonlyargs
        .iter()
        .chain(def_args.args.iter())
        .chain(def_args.kwonlyargs.iter())
        .chain(def_args.vararg.iter())
        .chain(def_args.kwarg.iter())
        .map(|p| p.arg.clone())
        .collect();
    walk_stmts(body, Descend::SkipDefs, &mut |s| {
        names.extend(crate::ast::tree::visit::stmt_bound_names(
            s,
            crate::ast::tree::visit::Bindings::Scope,
        ));
        Flow::Continue
    });
    names
}

/// The TYPE of a nested definition lowered as a callable value: its
/// annotated parameters and return, through the same annotation
/// authority every signature uses, so the binding in the enclosing scope
/// and the `PyCallable` the closure builds are the same type by
/// construction.
pub(crate) fn closure_typeinfo(
    def: &FunctionDef,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Option<crate::TypeInfo> {
    let mut params = Vec::new();
    for p in def.args.posonlyargs.iter().chain(def.args.args.iter()) {
        let ann = p.evaluated_annotation()?;
        params.push(crate::resolve_alias_typeinfo(&ann, symbols, options)?);
    }
    let returns = def.returns.as_deref()?;
    let ret = if crate::is_none_expr(returns) {
        crate::TypeInfo::Tuple(Vec::new())
    } else {
        crate::resolve_alias_typeinfo(returns, symbols, options)?
    };
    Some(crate::TypeInfo::Callable(params, Box::new(ret)))
}

/// The closure cell a statement MUTATES, if any — the same rule
/// [`closure_info`] uses to decide what a cell is, read from the other
/// side: a store through the name (`counter["n"] += 1`,
/// `counter["n"] = 0`) or a mutating method on it (`acc.append(x)`). A
/// bare-name target REBINDS the local and is the cell's own binding, not
/// a mutation of the object it holds.
///
/// The statement lowering borrows the cell mutably around such a
/// statement, so the ordinary store lowering writes the ONE object the
/// closure and the enclosing scope share.
pub(crate) fn cell_mutation_root(s: &Statement, cells: &HashSet<String>) -> Option<String> {
    for t in crate::ast::tree::visit::stmt_targets(s) {
        if matches!(t, ExprType::Name(_)) {
            continue;
        }
        if let Some(root) = store_root(t)
            && cells.contains(root)
        {
            return Some(root.to_string());
        }
    }
    let mut found: Option<String> = None;
    for e in crate::ast::tree::visit::stmt_all_exprs(s) {
        crate::ast::tree::visit::walk_expr(e, &mut |sub| {
            if found.is_some() {
                return;
            }
            if let ExprType::Call(c) = sub
                && let ExprType::Attribute(attr) = c.func.as_ref()
                && crate::ast::tree::scope::mutates_receiver(&attr.attr)
                && let ExprType::Name(recv) = attr.value.as_ref()
                && cells.contains(&recv.id)
            {
                found = Some(recv.id.clone());
            }
        });
    }
    found
}

/// The enclosing-scope names a LAMBDA captures: the names its body reads
/// that the scope binds and the lambda's own parameters do not. A lambda
/// is an expression, so the scope it captures from is only known through
/// the options the renderer carries — `name_types` is the scope's own
/// binding set (parameters and locals with a type), which is exactly the
/// set a `move` closure can clone.
pub(crate) fn lambda_captures(lam: &crate::Lambda, options: &PythonOptions) -> Vec<String> {
    let params: HashSet<&str> = lam
        .args
        .posonlyargs
        .iter()
        .chain(lam.args.args.iter())
        .chain(lam.args.kwonlyargs.iter())
        .map(|p| p.arg.as_str())
        .collect();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    crate::ast::tree::visit::walk_expr(&lam.body, &mut |sub| {
        if let ExprType::Name(n) = sub
            && !params.contains(n.id.as_str())
            && options.name_types.contains_key(&n.id)
            && seen.insert(n.id.clone())
        {
            out.push(n.id.clone());
        }
    });
    out
}

/// A `lambda` in callable-VALUE position (issue #122): the same
/// `stdpython::PyCallable` a nested `def` becomes, with its parameter
/// types taken from the type the position expects — a lambda writes no
/// annotations, so the expected `Callable[[A], R]` is what types it.
///
/// The body renders inside a `Result`-returning closure, so a fallible
/// expression (`x // 2`, a call that can raise) propagates its exception
/// to the CALLER through `?`, as a Python lambda's exception does —
/// where a lambda lowered as a plain Rust closure could only panic.
pub(crate) fn render_lambda_callable(
    lam: &crate::Lambda,
    params: &[TypeInfo],
    ret: &TypeInfo,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let names: Vec<&crate::Parameter> =
        lam.args.posonlyargs.iter().chain(lam.args.args.iter()).collect();
    if lam.args.vararg.is_some() || lam.args.kwarg.is_some() {
        return Err("a lambda taking `*args`/`**kwargs` cannot be a callable value: \
                    the argument tuple has a fixed arity"
            .to_string()
            .into());
    }
    if !lam.args.kwonlyargs.is_empty() {
        return Err("a lambda with a keyword-only parameter cannot be a callable \
                    value: the argument tuple is positional and has no names to \
                    match a keyword against"
            .to_string()
            .into());
    }
    if names.len() != params.len() {
        return Err(format!(
            "a lambda of {} parameter(s) cannot be the callable value \
             `{}` expects",
            names.len(),
            TypeInfo::Callable(params.to_vec(), Box::new(ret.clone())).display()
        )
        .into());
    }
    let idents: Vec<proc_macro2::Ident> =
        names.iter().map(|p| crate::safe_ident(&p.arg)).collect();
    let types: Vec<TokenStream> = params.iter().map(|t| t.to_rust_type()).collect();
    // A Rust 1-tuple needs its trailing comma; a 0-parameter lambda takes
    // the unit.
    let (pattern, arg_type) = if idents.len() == 1 {
        let (i, t) = (&idents[0], &types[0]);
        (quote!((#i,)), quote!((#t,)))
    } else {
        (quote!((#(#idents),*)), quote!((#(#types),*)))
    };
    let ret_type = ret.to_rust_type();
    // The lambda's parameters are LOCALS of its own scope with the types
    // the expected callable gives them, so the body lowers against them
    // (a `str` parameter is a String, an int stays i64).
    let mut inner = options.clone();
    let mut name_types = (*inner.name_types).clone();
    for (p, t) in names.iter().zip(params.iter()) {
        name_types.insert(p.arg.clone(), t.clone());
    }
    inner.name_types = std::rc::Rc::new(name_types);
    // A walrus in the body binds the lambda's own local, never the
    // enclosing statement's name.
    inner.stmt_binds = None;
    let captured: Vec<String> = lambda_captures(lam, &options)
        .into_iter()
        .filter(|c| !names.iter().any(|p| &p.arg == c))
        .collect();
    // A lambda used as a callable VALUE (a `move` closure in a
    // `PyCallable`) cannot capture the receiver of an enclosing METHOD:
    // its capture prologue is `let self = self.clone()` — E0424, `self`
    // may not be bound, and a `&mut` receiver has no clone — and the
    // value could outlive the method's borrow. A plain Rust closure
    // (an argument lambda) captures the receiver by reference instead;
    // this callable-value form must refuse loudly.
    if captured.iter().any(|c| c == "self") {
        return Err(format!(
            "a lambda used as a callable value cannot capture the method \
             receiver through `self`: the receiver is a borrow of the \
             instance, not a value to clone, and `self` may not be bound \
             in Rust. Pass the receiver's fields the lambda needs as \
             arguments"
        )
        .into());
    }
    // A lambda that MUTATES a captured container has no cell to write
    // through: cells are decided per STATEMENT (a nested `def`'s stores),
    // and a lambda is an expression whose captures are clones. Rather
    // than let the mutation vanish into the clone, refuse it here — the
    // nested `def` spelling carries it.
    for name in &captured {
        if crate::ast::tree::visit::any_expr(&lam.body, |sub| {
            matches!(sub, ExprType::Call(c)
                if matches!(c.func.as_ref(), ExprType::Attribute(attr)
                    if crate::ast::tree::scope::mutates_receiver(&attr.attr)
                        && matches!(attr.value.as_ref(), ExprType::Name(r) if &r.id == name)))
        }) {
            return Err(format!(
                "a lambda used as a callable value cannot mutate the captured \
                 `{name}`: its captures are clones, and only a nested `def`'s \
                 stores become the shared cell that Python's closure is \
                 (issue #122) — write it as a nested `def`"
            )
            .into());
        }
    }
    let captures: Vec<proc_macro2::Ident> =
        captured.iter().map(|c| crate::safe_ident(c)).collect();
    let body = crate::render_typed(
        &lam.body,
        ctx,
        inner,
        symbols,
        Some(ret.clone()),
    )?;
    Ok(quote! {{
        #(let #captures = #captures.clone();)*
        stdpython::PyCallable::new(
            "<lambda>",
            move |#pattern: #arg_type| -> Result<#ret_type, PyException> { Ok(#body) },
        )
    }})
}

/// Record the type a POSITION expects for every bare name that sits in
/// it, descending through container literals: an argument, a list
/// element, a dict value. `xs: list[Callable[[int], int]] = [double]`
/// says `double` is a `Callable[[int], int]` exactly as passing it to a
/// `Callable[[int], int]` parameter does.
fn expected_name_types(
    expr: &ExprType,
    expected: &TypeInfo,
    out: &mut Vec<(String, TypeInfo)>,
) {
    match (expr, expected) {
        (ExprType::Name(n), _) => out.push((n.id.clone(), expected.clone())),
        (ExprType::List(items), TypeInfo::Vec(inner)) => {
            for e in items {
                expected_name_types(e, inner, out);
            }
        }
        (ExprType::Tuple(t), TypeInfo::Vec(inner)) => {
            for e in &t.elts {
                expected_name_types(e, inner, out);
            }
        }
        (ExprType::Dict(d), TypeInfo::Dict(_, v)) => {
            for value in &d.values {
                expected_name_types(value, v, out);
            }
        }
        _ => {}
    }
}

/// What the uses of a name in `body` say its type is, for the positions
/// a type is written down in: an argument to a function of this module
/// whose parameter is annotated, and an annotated assignment.
fn uses_expect(
    body: &[Statement],
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Vec<(String, TypeInfo)> {
    let mut out = Vec::new();
    walk_stmts(body, Descend::All, &mut |s| {
        if let StatementType::Assign(a) = &s.statement
            && let Some(ann) = a.annotation.as_ref()
            && let Some(t) = crate::resolve_alias_typeinfo(ann, symbols, options)
        {
            expected_name_types(&a.value, &t, &mut out);
        }
        for e in crate::ast::tree::visit::stmt_all_exprs(s) {
            crate::ast::tree::visit::walk_expr(e, &mut |sub| {
                let ExprType::Call(c) = sub else { return };
                let ExprType::Name(callee) = c.func.as_ref() else {
                    return;
                };
                let Some(crate::SymbolTableNode::FunctionDef(def)) = symbols.get(&callee.id)
                else {
                    return;
                };
                let params: Vec<&crate::Parameter> = def
                    .args
                    .posonlyargs
                    .iter()
                    .chain(def.args.args.iter())
                    .collect();
                for (arg, p) in c.args.iter().zip(params.iter()) {
                    if let Some(ann) = p.evaluated_annotation()
                        && let Some(t) = crate::resolve_alias_typeinfo(&ann, symbols, options)
                    {
                        expected_name_types(arg, &t, &mut out);
                    }
                }
            });
        }
        Flow::Continue
    });
    out
}

/// The type of each local bound to a bare `lambda` (issue #122).
///
/// A Python lambda writes no annotations, so nothing about
/// `double = lambda x: x * 2` says what `x` is. What DOES say it is
/// where the name is used: `compose(add5, double)`, whose parameter is
/// `Callable[[int], int]`, or `ops: dict[str, Callable[[int], int]] =
/// {"double": double}`. Those positions are read here — the one place a
/// lambda local's type can come from. Uses that disagree name no type at
/// all (the definition is then loud at its use site) rather than picking
/// one.
pub(crate) fn lambda_local_types(
    body: &[Statement],
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Vec<(String, TypeInfo)> {
    let mut lambda_names: Vec<String> = Vec::new();
    walk_stmts(body, Descend::SkipDefs, &mut |s| {
        if let StatementType::Assign(a) = &s.statement
            && a.annotation.is_none()
            && a.targets.len() == 1
            && let ExprType::Name(t) = &a.targets[0]
            && matches!(a.value, ExprType::Lambda(_))
        {
            lambda_names.push(t.id.clone());
        }
        Flow::Continue
    });
    if lambda_names.is_empty() {
        return Vec::new();
    }
    let expectations = uses_expect(body, symbols, options);
    let mut out = Vec::new();
    for name in lambda_names {
        let mut candidates: Vec<&TypeInfo> = expectations
            .iter()
            .filter(|(n, t)| *n == name && matches!(t, TypeInfo::Callable(..)))
            .map(|(_, t)| t)
            .collect();
        candidates.dedup_by(|a, b| a == b);
        if candidates.len() == 1 {
            out.push((name, candidates[0].clone()));
        }
    }
    out
}

/// A module FUNCTION NAME used where a callable VALUE is expected
/// (`guarded(checked, 4)`, `_INITIALIZERS.append(callback)` — issue
/// #122): the name denotes a definition, not a value, so it is wrapped
/// in the one runtime callable type. The wrapper forwards to the item,
/// so the function keeps its own signature and its `Result` propagates
/// through the call exactly as a direct call's does.
///
/// `None` when the name is not a function of this module — a local
/// already holding a callable passes through unchanged.
pub(crate) fn wrap_function_as_callable(
    expr: &ExprType,
    params: &[TypeInfo],
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<Result<TokenStream, Box<dyn std::error::Error>>> {
    let ExprType::Name(n) = expr else {
        return None;
    };
    // A LOCAL of the same name shadows the definition and already holds
    // a callable value.
    if options.name_types.contains_key(&n.id) {
        return None;
    }
    let Some(crate::SymbolTableNode::FunctionDef(def)) = symbols.get(&n.id) else {
        return None;
    };
    // Keyword-only parameters have no place in a positional argument
    // tuple: `f(1)` would satisfy the wrapper where Python rejects it,
    // and `f(x=1)` could not be spelled at all.
    if !def.args.kwonlyargs.is_empty() {
        return Some(Err(format!(
            "`{}` has a keyword-only parameter and cannot be the callable value \
             this position expects: the argument tuple is positional",
            n.id
        )
        .into()));
    }
    let arity = def.args.posonlyargs.len() + def.args.args.len();
    if def.args.vararg.is_some() || def.args.kwarg.is_some() || arity != params.len() {
        return Some(Err(format!(
            "`{}` takes {}{} argument(s) and cannot be the callable value this \
             position expects, which takes {}",
            n.id,
            arity,
            if def.args.vararg.is_some() || def.args.kwarg.is_some() { " or more" } else { "" },
            params.len()
        )
        .into()));
    }
    let idents: Vec<proc_macro2::Ident> = (0..arity)
        .map(|i| quote::format_ident!("__rython_a{}", i))
        .collect();
    let types: Vec<TokenStream> = params.iter().map(|t| t.to_rust_type()).collect();
    let (pattern, arg_type) = if arity == 1 {
        let (i, t) = (&idents[0], &types[0]);
        (quote!((#i,)), quote!((#t,)))
    } else {
        (quote!((#(#idents),*)), quote!((#(#types),*)))
    };
    let item = crate::safe_ident(&n.id);
    let py_name = n.id.clone();
    Some(Ok(quote! {
        stdpython::PyCallable::new(
            #py_name,
            |#pattern: #arg_type| #item(#(#idents),*),
        )
    }))
}

/// Whether a captured name CANNOT change after the definitions in this
/// scope run — the one case where cloning it into a closure is
/// indistinguishable from sharing its cell (see the module docs).
///
/// It must be bound exactly once, and that binding must be
/// unconditional-in-time: a parameter, or a top-level statement outside
/// every loop. A binding inside a loop rebinds on each turn (every
/// closure built there shares the last value in Python), and a second
/// binding anywhere means a later value the closure would have to see.
/// It must also be mutated by nobody — the enclosing scope or any nested
/// definition — since a mutation changes the object the name is bound to.
fn capture_cannot_change(
    name: &str,
    args: &crate::ParameterList,
    body: &[Statement],
) -> bool {
    let mut bindings = usize::from(
        args.posonlyargs
            .iter()
            .chain(args.args.iter())
            .chain(args.kwonlyargs.iter())
            .chain(args.vararg.iter())
            .chain(args.kwarg.iter())
            .any(|p| p.arg == name),
    );
    let mut rebound_under_a_loop = false;
    count_bindings(body, name, false, &mut bindings, &mut rebound_under_a_loop);
    if bindings != 1 || rebound_under_a_loop {
        return false;
    }
    // A mutation ANYWHERE in the scope, nested definitions and lambda
    // bodies included: the closure must see it.
    !mutates_name(body, name)
}

/// Count the statements of `body` that bind `name`, flagging any that sit
/// inside a loop. Nested definitions are their own scopes and do not
/// bind this one's name.
fn count_bindings(
    body: &[Statement],
    name: &str,
    in_loop: bool,
    bindings: &mut usize,
    under_loop: &mut bool,
) {
    for s in body {
        let loops = matches!(
            &s.statement,
            StatementType::For(_) | StatementType::AsyncFor(_) | StatementType::While(_)
        );
        if crate::ast::tree::visit::stmt_bound_names(
            s,
            crate::ast::tree::visit::Bindings::Scope,
        )
        .iter()
        .any(|n| n == name)
        {
            *bindings += 1;
            // A `for` TARGET is rebound on every turn, so one syntactic
            // binding is many bindings in time: in Python each closure
            // built in the loop sees the LAST value.
            if in_loop
                || matches!(&s.statement, StatementType::For(_) | StatementType::AsyncFor(_))
            {
                *under_loop = true;
            }
        }
        for inner in crate::ast::tree::visit::stmt_bodies_for(s, Descend::SkipDefs) {
            count_bindings(inner, name, in_loop || loops, bindings, under_loop);
        }
    }
}

/// Whether anything in `body` — this scope, a nested definition, or a
/// lambda — mutates the object `name` is bound to: a store through it, or
/// a mutating method on it.
fn mutates_name(body: &[Statement], name: &str) -> bool {
    let mut found = false;
    walk_stmts(body, Descend::All, &mut |s| {
        for t in crate::ast::tree::visit::stmt_targets(s) {
            if !matches!(t, ExprType::Name(_)) && store_root(t) == Some(name) {
                found = true;
            }
        }
        for e in crate::ast::tree::visit::stmt_all_exprs(s) {
            crate::ast::tree::visit::walk_expr(e, &mut |sub| {
                if let ExprType::Call(c) = sub
                    && let ExprType::Attribute(attr) = c.func.as_ref()
                    && crate::ast::tree::scope::mutates_receiver(&attr.attr)
                    && matches!(attr.value.as_ref(), ExprType::Name(r) if r.id == name)
                {
                    found = true;
                }
            });
        }
        if found { Flow::Stop } else { Flow::Continue }
    });
    found
}

/// Every name a nested definition or a callable-position lambda in this
/// scope captures, and which of them must be CELLS.
///
/// The captures come from the definitions' free names; a name is a cell
/// unless [`capture_cannot_change`] proves the clone is
/// indistinguishable from the cell.
pub(crate) fn scope_cell_locals(
    args: &crate::ParameterList,
    body: &[Statement],
    scope_names: &HashSet<String>,
) -> HashSet<String> {
    let mut candidates: HashSet<String> = HashSet::new();
    // A nested `def`'s free names.
    walk_stmts(body, Descend::SkipDefs, &mut |s| {
        if let StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) =
            &s.statement
        {
            candidates.extend(closure_info(f, scope_names).captures);
        }
        Flow::Continue
    });
    // A LAMBDA's free names: a lambda in a callable position captures the
    // same way, and one anywhere else is a plain Rust closure that reads
    // the binding in place — a cell serves both.
    walk_stmts(body, Descend::SkipDefs, &mut |s| {
        for e in crate::ast::tree::visit::stmt_all_exprs(s) {
            crate::ast::tree::visit::walk_expr(e, &mut |sub| {
                if let ExprType::Lambda(lam) = sub {
                    let params: HashSet<&str> = lam
                        .args
                        .posonlyargs
                        .iter()
                        .chain(lam.args.args.iter())
                        .chain(lam.args.kwonlyargs.iter())
                        .map(|p| p.arg.as_str())
                        .collect();
                    crate::ast::tree::visit::walk_expr(&lam.body, &mut |inner| {
                        if let ExprType::Name(n) = inner
                            && !params.contains(n.id.as_str())
                            && scope_names.contains(&n.id)
                        {
                            candidates.insert(n.id.clone());
                        }
                    });
                }
            });
        }
        Flow::Continue
    });
    candidates
        .into_iter()
        .filter(|name| !capture_cannot_change(name, args, body))
        .collect()
}

/// Whether a nested definition's HEADER runs code at the `def` — a
/// decorator, or a default whose expression is not a literal.
///
/// Python evaluates these WHERE THE `def` STANDS, so a definition the
/// closure model refuses cannot simply vanish: its header's output,
/// mutations and exceptions are part of the program. Such a definition
/// is a conversion error even when the name is never read or called.
pub(crate) fn header_runs_code(def: &FunctionDef) -> Option<String> {
    if !def.decorator_list.is_empty() {
        return Some(format!(
            "nested function `{}` has a decorator, which Python EVALUATES where \
             the `def` stands — dropping the definition would drop that too. \
             Move the definition to module level, or apply the decorator \
             explicitly to a value the closure model can carry",
            def.name
        ));
    }
    let evaluates = |e: &ExprType| {
        !matches!(e, ExprType::Constant(_) | ExprType::NoneType(_))
    };
    if def.args.defaults.iter().any(|d| evaluates(d))
        || def
            .args
            .kw_defaults
            .iter()
            .flatten()
            .any(|d| evaluates(d))
    {
        return Some(format!(
            "nested function `{}` has a default whose expression Python \
             EVALUATES where the `def` stands — dropping the definition would \
             drop that too. Bind the default to a local before the `def` and \
             pass it explicitly",
            def.name
        ));
    }
    None
}
