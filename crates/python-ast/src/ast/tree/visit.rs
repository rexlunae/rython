//! The one statement and expression visitor (issue #137, drift 4).
//!
//! Every analysis that walks a Python body asks these enumerations
//! instead of matching the control-flow forms itself, so no walker can
//! miss an `async for`, an `else` clause, or a `finally` block, and a new
//! statement form is added in one place.

use crate::{ExprType, Statement, StatementType};

/// Whether a walk enters the bodies of nested `def`s and `class`es (their
/// own scopes) or stays in the scope it started in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Descend {
    /// Every body, nested definitions included.
    All,
    /// Control-flow bodies only; a nested `def` / `async def` / `class`
    /// is visited as a statement but its body is not entered.
    SkipDefs,
}

/// What the callback asks of the walk after seeing a statement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flow {
    /// Descend into this statement's bodies.
    Continue,
    /// Do not enter this statement's bodies.
    Skip,
    /// End the whole walk.
    Stop,
}

/// Whether an expression is the bare receiver `self`.
pub fn is_self(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Name(n) if n.id == "self")
}

/// Whether a statement opens its own scope (a `def`, an `async def`, a
/// `class`).
pub fn opens_scope(s: &Statement) -> bool {
    matches!(
        s.statement,
        StatementType::FunctionDef(_) | StatementType::AsyncFunctionDef(_) | StatementType::ClassDef(_)
    )
}

/// The nested statement bodies of a statement, every control-flow form
/// (the asynchronous ones included).
pub fn stmt_bodies(s: &Statement) -> Vec<&[Statement]> {
    match &s.statement {
        StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => vec![&f.body],
        StatementType::ClassDef(c) => vec![&c.body],
        StatementType::If(i) => vec![&i.body, &i.orelse],
        StatementType::For(f) => vec![&f.body, &f.orelse],
        StatementType::AsyncFor(f) => vec![&f.body, &f.orelse],
        StatementType::While(w) => vec![&w.body, &w.orelse],
        StatementType::With(w) => vec![&w.body],
        StatementType::AsyncWith(w) => vec![&w.body],
        StatementType::Try(t) => std::iter::once(t.body.as_slice())
            .chain(t.handlers.iter().map(|h| h.body.as_slice()))
            .chain([t.orelse.as_slice(), t.finalbody.as_slice()])
            .collect(),
        _ => Vec::new(),
    }
}


/// The nested statement bodies a walk with `descend` enters.
pub fn stmt_bodies_for(s: &Statement, descend: Descend) -> Vec<&[Statement]> {
    if descend == Descend::SkipDefs && opens_scope(s) {
        return Vec::new();
    }
    stmt_bodies(s)
}

/// The expressions a statement evaluates itself (its bodies aside): the
/// test of an `if` / `while`, the iterable of a `for`, a `with` item's
/// context, an assert, a raise, a return, an expression statement, a
/// store's value — so a mutation in any of them is a mutation.
pub fn stmt_exprs(s: &Statement) -> Vec<&ExprType> {
    match &s.statement {
        StatementType::Assign(a) => vec![&a.value],
        StatementType::AugAssign(a) => vec![&a.value],
        StatementType::Expr(e) => vec![&e.value],
        StatementType::Return(Some(e)) => vec![&e.value],
        StatementType::If(i) => vec![&i.test],
        StatementType::While(w) => vec![&w.test],
        StatementType::For(f) => vec![&f.iter],
        StatementType::AsyncFor(f) => vec![&f.iter],
        StatementType::With(w) => w.items.iter().map(|i| &i.context_expr).collect(),
        StatementType::AsyncWith(w) => w.items.iter().map(|i| &i.context_expr).collect(),
        StatementType::Assert { test, msg } => {
            std::iter::once(test.as_ref()).chain(msg.iter().map(|m| m.as_ref())).collect()
        }
        StatementType::Raise(r) => r.exc.iter().chain(r.cause.iter()).collect(),
        StatementType::Delete(targets) => targets.iter().collect(),
        _ => Vec::new(),
    }
}


/// The binding targets a statement writes as EXPRESSIONS (an assignment's
/// targets, an augmented assignment's, a `for` / `async for` target, a
/// `with` item's `as`), for the walks that look at where a value lands.
pub fn stmt_targets(s: &Statement) -> Vec<&ExprType> {
    match &s.statement {
        StatementType::Assign(a) => a.targets.iter().collect(),
        StatementType::AugAssign(a) => vec![&a.target],
        StatementType::For(f) => vec![&f.target],
        StatementType::AsyncFor(f) => vec![&f.target],
        StatementType::With(w) => w.items.iter().filter_map(|i| i.optional_vars.as_ref()).collect(),
        StatementType::AsyncWith(w) => w.items.iter().filter_map(|i| i.optional_vars.as_ref()).collect(),
        _ => Vec::new(),
    }
}

/// Whether a binding target (a store's target, a `for` target, a `with
/// ... as`) binds `name`: a bare name, or one inside a tuple / list /
/// starred pattern. An attribute or subscript target mutates an object
/// and binds no name.
pub fn target_binds(target: &ExprType, name: &str) -> bool {
    match target {
        ExprType::Name(n) => n.id == name,
        ExprType::Tuple(t) => t.elts.iter().any(|e| target_binds(e, name)),
        ExprType::List(items) => items.iter().any(|e| target_binds(e, name)),
        ExprType::Starred(st) => target_binds(&st.value, name),
        _ => false,
    }
}

/// Whether a nested function binds `name` in its OWN scope — a parameter,
/// or a store anywhere in its body (an assignment, an augmented
/// assignment, a loop or `with ... as` target, an `except ... as`, an
/// import, a nested def or class of that name) — and does not declare it
/// `nonlocal` / `global`. A name the function does not bind is FREE in
/// it: it refers to the enclosing scope's binding, so a mutation through
/// it mutates the enclosing value and an `isinstance` on it tests the
/// enclosing parameter (Devin review on #323).
pub fn def_binds_locally(f: &crate::FunctionDef, name: &str) -> bool {
    let args = &f.args;
    let mut params = args
        .posonlyargs
        .iter()
        .chain(args.args.iter())
        .chain(args.kwonlyargs.iter())
        .chain(args.vararg.iter())
        .chain(args.kwarg.iter());
    if params.any(|p| p.arg == name) {
        return true;
    }
    let declared_free = any_stmt(&f.body, Descend::SkipDefs, |s| {
        matches!(&s.statement, StatementType::Global(names) | StatementType::Nonlocal(names)
            if names.iter().any(|n| n == name))
    });
    if declared_free {
        return false;
    }
    any_stmt(&f.body, Descend::SkipDefs, |s| {
        stmt_targets(s).into_iter().any(|t| target_binds(t, name))
            || match &s.statement {
                StatementType::Import(im) => im.names.iter().any(|a| match a.asname.as_deref() {
                    Some(asname) => asname == name,
                    None => a.name.split('.').next() == Some(name),
                }),
                StatementType::ImportFrom(im) => im
                    .names
                    .iter()
                    .any(|a| a.asname.as_deref().unwrap_or(&a.name) == name),
                StatementType::Try(t) => t.handlers.iter().any(|h| h.name.as_deref() == Some(name)),
                StatementType::FunctionDef(d) | StatementType::AsyncFunctionDef(d) => d.name == name,
                StatementType::ClassDef(c) => c.name == name,
                _ => false,
            }
    })
}

/// The direct subexpressions of an expression: one enumeration for every
/// walk in this module, so a call nested anywhere (`x = 1 + self.q.pop()`,
/// `if self.items.pop():`) is seen. A comprehension's and a lambda's body
/// are included (they run in the method).
pub fn subexprs(e: &ExprType) -> Vec<&ExprType> {
    match e {
        ExprType::BoolOp(b) => b.values.iter().collect(),
        ExprType::NamedExpr(n) => vec![&n.left, &n.right],
        ExprType::BinOp(b) => vec![&b.left, &b.right],
        ExprType::UnaryOp(u) => vec![&u.operand],
        ExprType::Lambda(l) => vec![&l.body],
        ExprType::IfExp(i) => vec![&i.test, &i.body, &i.orelse],
        ExprType::Dict(d) => d.keys.iter().flatten().chain(d.values.iter()).collect(),
        ExprType::Set(s) => s.elts.iter().collect(),
        ExprType::ListComp(c) => std::iter::once(c.elt.as_ref())
            .chain(c.generators.iter().flat_map(|g| {
                std::iter::once(&g.iter).chain(g.ifs.iter())
            }))
            .collect(),
        ExprType::SetComp(c) => std::iter::once(c.elt.as_ref())
            .chain(c.generators.iter().flat_map(|g| {
                std::iter::once(&g.iter).chain(g.ifs.iter())
            }))
            .collect(),
        ExprType::GeneratorExp(c) => std::iter::once(c.elt.as_ref())
            .chain(c.generators.iter().flat_map(|g| {
                std::iter::once(&g.iter).chain(g.ifs.iter())
            }))
            .collect(),
        ExprType::DictComp(c) => [c.key.as_ref(), c.value.as_ref()]
            .into_iter()
            .chain(c.generators.iter().flat_map(|g| {
                std::iter::once(&g.iter).chain(g.ifs.iter())
            }))
            .collect(),
        ExprType::Await(a) => vec![&a.value],
        ExprType::Yield(y) => y.value.iter().map(|v| v.as_ref()).collect(),
        ExprType::YieldFrom(y) => vec![&y.value],
        ExprType::Compare(c) => std::iter::once(c.left.as_ref()).chain(c.comparators.iter()).collect(),
        ExprType::Call(c) => std::iter::once(c.func.as_ref())
            .chain(c.args.iter())
            .chain(c.keywords.iter().map(|k| &k.value))
            .collect(),
        ExprType::FormattedValue(f) => std::iter::once(f.value.as_ref())
            .chain(f.format_spec.iter().map(|spec| spec.as_ref()))
            .collect(),
        ExprType::JoinedStr(j) => j.values.iter().collect(),
        ExprType::Attribute(a) => vec![&a.value],
        ExprType::Subscript(s) => {
            let mut out = vec![s.value.as_ref()];
            match &s.kind {
                crate::SubscriptKind::Index(i) => out.push(i),
                crate::SubscriptKind::Slice { lower, upper, step } => {
                    out.extend(lower.iter().chain(upper.iter()).chain(step.iter()).map(|b| b.as_ref()));
                }
            }
            out
        }
        ExprType::Starred(st) => vec![&st.value],
        ExprType::List(l) => l.iter().collect(),
        ExprType::Tuple(t) => t.elts.iter().collect(),
        _ => Vec::new(),
    }
}


/// Walk `stmts` in source order, pre-order, calling `f` on every statement
/// and entering the bodies it allows. Returns false when `f` stopped the
/// walk.
pub fn walk_stmts<'a>(
    stmts: &'a [Statement],
    descend: Descend,
    f: &mut impl FnMut(&'a Statement) -> Flow,
) -> bool {
    for s in stmts {
        match f(s) {
            Flow::Stop => return false,
            Flow::Skip => continue,
            Flow::Continue => {
                for body in stmt_bodies_for(s, descend) {
                    if !walk_stmts(body, descend, f) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Whether any statement of `stmts` (bodies per `descend`) satisfies
/// `pred`.
pub fn any_stmt<'a>(
    stmts: &'a [Statement],
    descend: Descend,
    mut pred: impl FnMut(&'a Statement) -> bool,
) -> bool {
    !walk_stmts(stmts, descend, &mut |s| if pred(s) { Flow::Stop } else { Flow::Continue })
}

/// Walk an expression and every subexpression, pre-order.
pub fn walk_expr<'a>(e: &'a ExprType, f: &mut impl FnMut(&'a ExprType)) {
    f(e);
    for sub in subexprs(e) {
        walk_expr(sub, f);
    }
}

/// Whether the expression or any subexpression satisfies `pred`.
pub fn any_expr<'a>(e: &'a ExprType, mut pred: impl FnMut(&'a ExprType) -> bool) -> bool {
    fn go<'a>(e: &'a ExprType, pred: &mut impl FnMut(&'a ExprType) -> bool) -> bool {
        pred(e) || subexprs(e).into_iter().any(|sub| go(sub, pred))
    }
    go(e, &mut pred)
}

/// Every expression a statement evaluates or binds, with the statement's
/// bodies left to `walk_stmts`: the evaluated expressions and the targets.
pub fn stmt_all_exprs(s: &Statement) -> Vec<&ExprType> {
    let mut out = stmt_exprs(s);
    out.extend(stmt_targets(s));
    out
}

/// Whether any expression in `stmts` (the statements' own expressions and
/// their subexpressions, bodies per `descend`) satisfies `pred`.
pub fn any_expr_in<'a>(
    stmts: &'a [Statement],
    descend: Descend,
    mut pred: impl FnMut(&'a ExprType) -> bool,
) -> bool {
    any_stmt(stmts, descend, |s| stmt_all_exprs(s).into_iter().any(|e| any_expr(e, &mut pred)))
}
