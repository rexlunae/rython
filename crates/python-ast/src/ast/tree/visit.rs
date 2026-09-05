//! The one statement and expression visitor (issue #137, drift 4).
//!
//! Every analysis that walks a Python body asks these enumerations
//! instead of matching the control-flow forms itself, so no walker can
//! miss an `async for`, an `else` clause, or a `finally` block, and a new
//! statement form is added in one place.

use crate::{ExprType, Statement, StatementType};

/// Which nested scopes a walk enters: the one scope policy for statement
/// bodies AND expression bodies (a lambda's), so no analysis filters
/// scopes on its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Descend {
    /// Every body, nested definitions and lambda bodies included.
    All,
    /// Control-flow bodies only; a nested `def` / `async def` / `class`
    /// is visited as a statement but its body is not entered. A LAMBDA's
    /// body is entered: it is a closure over this scope, so what it
    /// reads, mutates, or calls is this scope's business.
    SkipDefs,
    /// The scope's own code only: nested definitions AND lambda bodies
    /// stay out. For what the function itself does — its `yield`s (a
    /// yielding lambda is its own generator), its `await`s.
    OwnScope,
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
    if descend != Descend::All && opens_scope(s) {
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
        // A definition's HEADER runs in this scope when the definition
        // does: decorators, defaults, annotations, bases, keywords. The
        // body is the definition's own scope (`stmt_bodies`).
        StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => def_header_exprs(f),
        StatementType::ClassDef(c) => c
            .decorator_list
            .iter()
            .chain(c.bases.iter())
            .chain(c.keywords.iter().map(|k| &k.value))
            .collect(),
        _ => Vec::new(),
    }
}

/// The expressions a `def`'s header evaluates in the ENCLOSING scope:
/// decorators, parameter defaults, parameter and return annotations.
pub fn def_header_exprs(f: &crate::FunctionDef) -> Vec<&ExprType> {
    let args = &f.args;
    let params = args
        .posonlyargs
        .iter()
        .chain(args.args.iter())
        .chain(args.kwonlyargs.iter())
        .chain(args.vararg.iter())
        .chain(args.kwarg.iter());
    f.decorator_list
        .iter()
        .chain(args.defaults.iter().map(|d| d.as_ref()))
        .chain(args.kw_defaults.iter().flatten().map(|d| d.as_ref()))
        .chain(params.filter_map(|p| p.annotation.as_deref()))
        .chain(f.returns.as_deref())
        .collect()
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

/// The names a binding target binds: a name, every element of a tuple or
/// list pattern, a starred element's name.
pub fn target_names(target: &ExprType) -> Vec<&str> {
    match target {
        ExprType::Name(n) => vec![n.id.as_str()],
        ExprType::Tuple(t) => t.elts.iter().flat_map(target_names).collect(),
        ExprType::List(items) => items.iter().flat_map(target_names).collect(),
        ExprType::Starred(st) => target_names(&st.value),
        _ => Vec::new(),
    }
}

/// The prefix of every temporary the conversion emits into the generated
/// crate (`__rython_load`, `__rython_recv`, `__rython_exc_arg0`, ...).
pub const RESERVED_PREFIX: &str = "__rython_";

/// The first binding in `body` (nested definitions included) of a name
/// under [`RESERVED_PREFIX`], with its line: a store target, a loop or
/// `with ... as` target, a walrus, a function or class name, a parameter,
/// an import, an `except ... as`, a `global`/`nonlocal` declaration. Such
/// a name would be shadowed by, or shadow, the conversion's own
/// temporaries silently (Devin review on #331), so the module refuses it
/// loudly.
pub fn reserved_prefix_binding(body: &[Statement]) -> Option<(String, usize)> {
    let mut found: Option<(String, usize)> = None;
    let reserved = |n: &str| n.starts_with(RESERVED_PREFIX);
    walk_stmts(body, Descend::All, &mut |s| {
        let line = s.lineno.unwrap_or(0);
        let mut names: Vec<String> = Vec::new();
        for t in stmt_targets(s) {
            names.extend(target_names(t).into_iter().map(str::to_string));
        }
        match &s.statement {
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                names.push(f.name.clone());
                let a = &f.args;
                names.extend(
                    a.posonlyargs
                        .iter()
                        .chain(a.args.iter())
                        .chain(a.kwonlyargs.iter())
                        .chain(a.vararg.iter())
                        .chain(a.kwarg.iter())
                        .map(|p| p.arg.clone()),
                );
            }
            StatementType::ClassDef(c) => names.push(c.name.clone()),
            StatementType::Import(im) => names.extend(im.names.iter().map(|a| match &a.asname {
                Some(asname) => asname.clone(),
                None => a.name.split('.').next().unwrap_or(&a.name).to_string(),
            })),
            StatementType::ImportFrom(im) => names.extend(
                im.names.iter().map(|a| a.asname.clone().unwrap_or_else(|| a.name.clone())),
            ),
            StatementType::Try(t) => names.extend(t.handlers.iter().filter_map(|h| h.name.clone())),
            StatementType::Global(ns) | StatementType::Nonlocal(ns) => names.extend(ns.iter().cloned()),
            _ => {}
        }
        for e in stmt_exprs(s) {
            walk_expr(e, &mut |x| {
                if let ExprType::NamedExpr(ne) = x {
                    names.extend(target_names(&ne.left).into_iter().map(str::to_string));
                }
            });
        }
        if let Some(n) = names.into_iter().find(|n| reserved(n)) {
            found = Some((n, line));
            return Flow::Stop;
        }
        Flow::Continue
    });
    found
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

/// Whether, inside the nested function `f`, `name` means something OTHER
/// than the enclosing scope's binding: a parameter, a store anywhere in
/// its body (an assignment, an augmented assignment, a loop or `with ...
/// as` target, an `except ... as`, an import, a nested def or class of
/// that name), or a `global` declaration (the module's binding, not the
/// enclosing function's). A `nonlocal` declaration is the enclosing
/// binding by definition, and a name `f` neither binds nor declares is
/// FREE in it: it refers to the enclosing scope's binding, so a mutation
/// through it mutates the enclosing value and an `isinstance` on it
/// tests the enclosing parameter (Devin review on #323).
pub fn def_owns_name(f: &crate::FunctionDef, name: &str) -> bool {
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
    let declares = |names: &[String]| names.iter().any(|n| n == name);
    if any_stmt(&f.body, Descend::SkipDefs, |s| {
        matches!(&s.statement, StatementType::Nonlocal(names) if declares(names))
    }) {
        return false;
    }
    any_stmt(&f.body, Descend::SkipDefs, |s| {
        stmt_targets(s).into_iter().any(|t| target_binds(t, name))
            || match &s.statement {
                StatementType::Global(names) => declares(names),
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

/// Call `f` on each direct subexpression of `e` that a walk with
/// `descend` enters, in source order, until `f` returns false; returns
/// false when stopped. The ONE enumeration behind `subexprs`,
/// `walk_expr`, and `any_expr`, with no allocation per node — so a call
/// nested anywhere (`x = 1 + self.q.pop()`, `if self.items.pop():`) is
/// seen. A comprehension's body is included (it runs in the scope); a
/// lambda's body is its own scope that only `Descend::OwnScope` stays
/// out of.
pub fn each_subexpr<'a>(
    e: &'a ExprType,
    descend: Descend,
    f: &mut impl FnMut(&'a ExprType) -> bool,
) -> bool {
    macro_rules! visit {
        ($x:expr) => {{
            let sub: &'a ExprType = &$x;
            if !f(sub) {
                return false;
            }
        }};
    }
    match e {
        ExprType::BoolOp(b) => {
            for v in &b.values {
                visit!(v);
            }
        }
        ExprType::NamedExpr(n) => {
            visit!(n.left);
            visit!(n.right);
        }
        ExprType::BinOp(b) => {
            visit!(b.left);
            visit!(b.right);
        }
        ExprType::UnaryOp(u) => visit!(u.operand),
        ExprType::Lambda(l) => {
            if descend != Descend::OwnScope {
                visit!(l.body);
            }
        }
        ExprType::IfExp(i) => {
            visit!(i.test);
            visit!(i.body);
            visit!(i.orelse);
        }
        ExprType::Dict(d) => {
            for k in d.keys.iter().flatten() {
                visit!(k);
            }
            for v in &d.values {
                visit!(v);
            }
        }
        ExprType::Set(s) => {
            for x in &s.elts {
                visit!(x);
            }
        }
        ExprType::ListComp(c) => {
            visit!(c.elt);
            for g in &c.generators {
                visit!(g.iter);
                for i in &g.ifs {
                    visit!(i);
                }
            }
        }
        ExprType::SetComp(c) => {
            visit!(c.elt);
            for g in &c.generators {
                visit!(g.iter);
                for i in &g.ifs {
                    visit!(i);
                }
            }
        }
        ExprType::GeneratorExp(c) => {
            visit!(c.elt);
            for g in &c.generators {
                visit!(g.iter);
                for i in &g.ifs {
                    visit!(i);
                }
            }
        }
        ExprType::DictComp(c) => {
            visit!(c.key);
            visit!(c.value);
            for g in &c.generators {
                visit!(g.iter);
                for i in &g.ifs {
                    visit!(i);
                }
            }
        }
        ExprType::Await(a) => visit!(a.value),
        ExprType::Yield(y) => {
            if let Some(v) = &y.value {
                visit!(v);
            }
        }
        ExprType::YieldFrom(y) => visit!(y.value),
        ExprType::Compare(c) => {
            visit!(c.left);
            for x in &c.comparators {
                visit!(x);
            }
        }
        ExprType::Call(c) => {
            visit!(c.func);
            for a in &c.args {
                visit!(a);
            }
            for k in &c.keywords {
                visit!(k.value);
            }
        }
        ExprType::FormattedValue(fv) => {
            visit!(fv.value);
            if let Some(spec) = &fv.format_spec {
                visit!(spec);
            }
        }
        ExprType::JoinedStr(j) => {
            for v in &j.values {
                visit!(v);
            }
        }
        ExprType::Attribute(a) => visit!(a.value),
        ExprType::Subscript(sub) => {
            visit!(sub.value);
            match &sub.kind {
                crate::SubscriptKind::Index(i) => visit!(i),
                crate::SubscriptKind::Slice { lower, upper, step } => {
                    for b in lower.iter().chain(upper.iter()).chain(step.iter()) {
                        visit!(b);
                    }
                }
            }
        }
        ExprType::Starred(st) => visit!(st.value),
        ExprType::List(l) => {
            for x in l {
                visit!(x);
            }
        }
        ExprType::Tuple(t) => {
            for x in &t.elts {
                visit!(x);
            }
        }
        _ => {}
    }
    true
}

/// The MUTABLE twin of `each_subexpr` — the same enumeration, arm for
/// arm (kept adjacent so a new expression form is added to both), for
/// the rewrites that replace nodes in place (the exception model's
/// parameter substitution). Lambda bodies are entered per `descend`.
pub fn each_subexpr_mut<'a>(
    e: &'a mut ExprType,
    descend: Descend,
    f: &mut impl FnMut(&'a mut ExprType) -> bool,
) -> bool {
    macro_rules! visit {
        ($x:expr) => {{
            let sub: &'a mut ExprType = &mut $x;
            if !f(sub) {
                return false;
            }
        }};
    }
    match e {
        ExprType::BoolOp(b) => {
            for v in &mut b.values {
                visit!(*v);
            }
        }
        ExprType::NamedExpr(n) => {
            visit!(n.left);
            visit!(n.right);
        }
        ExprType::BinOp(b) => {
            visit!(b.left);
            visit!(b.right);
        }
        ExprType::UnaryOp(u) => visit!(u.operand),
        ExprType::Lambda(l) => {
            if descend != Descend::OwnScope {
                visit!(l.body);
            }
        }
        ExprType::IfExp(i) => {
            visit!(i.test);
            visit!(i.body);
            visit!(i.orelse);
        }
        ExprType::Dict(d) => {
            for k in d.keys.iter_mut().flatten() {
                visit!(*k);
            }
            for v in &mut d.values {
                visit!(*v);
            }
        }
        ExprType::Set(s) => {
            for x in &mut s.elts {
                visit!(*x);
            }
        }
        ExprType::ListComp(c) => {
            visit!(c.elt);
            for g in &mut c.generators {
                visit!(g.iter);
                for i in &mut g.ifs {
                    visit!(*i);
                }
            }
        }
        ExprType::SetComp(c) => {
            visit!(c.elt);
            for g in &mut c.generators {
                visit!(g.iter);
                for i in &mut g.ifs {
                    visit!(*i);
                }
            }
        }
        ExprType::GeneratorExp(c) => {
            visit!(c.elt);
            for g in &mut c.generators {
                visit!(g.iter);
                for i in &mut g.ifs {
                    visit!(*i);
                }
            }
        }
        ExprType::DictComp(c) => {
            visit!(c.key);
            visit!(c.value);
            for g in &mut c.generators {
                visit!(g.iter);
                for i in &mut g.ifs {
                    visit!(*i);
                }
            }
        }
        ExprType::Await(a) => visit!(a.value),
        ExprType::Yield(y) => {
            if let Some(v) = &mut y.value {
                visit!(**v);
            }
        }
        ExprType::YieldFrom(y) => visit!(y.value),
        ExprType::Compare(c) => {
            visit!(c.left);
            for x in &mut c.comparators {
                visit!(*x);
            }
        }
        ExprType::Call(c) => {
            visit!(c.func);
            for a in &mut c.args {
                visit!(*a);
            }
            for k in &mut c.keywords {
                visit!(k.value);
            }
        }
        ExprType::FormattedValue(fv) => {
            visit!(fv.value);
            if let Some(spec) = &mut fv.format_spec {
                visit!(**spec);
            }
        }
        ExprType::JoinedStr(j) => {
            for v in &mut j.values {
                visit!(*v);
            }
        }
        ExprType::Attribute(a) => visit!(a.value),
        ExprType::Subscript(sub) => {
            visit!(sub.value);
            match &mut sub.kind {
                crate::SubscriptKind::Index(i) => visit!(*i),
                crate::SubscriptKind::Slice { lower, upper, step } => {
                    for b in lower.iter_mut().chain(upper.iter_mut()).chain(step.iter_mut()) {
                        visit!(**b);
                    }
                }
            }
        }
        ExprType::Starred(st) => visit!(st.value),
        ExprType::List(l) => {
            for x in l.iter_mut() {
                visit!(*x);
            }
        }
        ExprType::Tuple(t) => {
            for x in &mut t.elts {
                visit!(*x);
            }
        }
        _ => {}
    }
    true
}

/// Rewrite an expression in place, pre-order: `f` sees each node (and
/// may replace it), then the walk descends into the node's — possibly
/// new — subexpressions. Lambda bodies are entered (`Descend::All`).
pub fn walk_expr_mut(e: &mut ExprType, f: &mut impl FnMut(&mut ExprType)) {
    f(e);
    each_subexpr_mut(e, Descend::All, &mut |sub| {
        walk_expr_mut(sub, f);
        true
    });
}

/// The direct subexpressions of an expression, collected (lambda bodies
/// included: `Descend::All`).
pub fn subexprs(e: &ExprType) -> Vec<&ExprType> {
    subexprs_for(e, Descend::All)
}

/// The direct subexpressions a walk with `descend` enters, collected.
pub fn subexprs_for(e: &ExprType, descend: Descend) -> Vec<&ExprType> {
    let mut out = Vec::new();
    each_subexpr(e, descend, &mut |sub| {
        out.push(sub);
        true
    });
    out
}

/// Walk an expression and every subexpression, pre-order (lambda bodies
/// included: `Descend::All`).
pub fn walk_expr<'a>(e: &'a ExprType, f: &mut impl FnMut(&'a ExprType)) {
    f(e);
    each_subexpr(e, Descend::All, &mut |sub| {
        walk_expr(sub, f);
        true
    });
}

/// Whether the expression or any subexpression satisfies `pred` (lambda
/// bodies included: `Descend::All`).
pub fn any_expr<'a>(e: &'a ExprType, pred: impl FnMut(&'a ExprType) -> bool) -> bool {
    any_expr_for(e, Descend::All, pred)
}

/// Whether the expression or any subexpression a walk with `descend`
/// enters satisfies `pred`; stops at the first hit.
pub fn any_expr_for<'a>(
    e: &'a ExprType,
    descend: Descend,
    mut pred: impl FnMut(&'a ExprType) -> bool,
) -> bool {
    fn go<'a>(e: &'a ExprType, descend: Descend, pred: &mut impl FnMut(&'a ExprType) -> bool) -> bool {
        if pred(e) {
            return true;
        }
        !each_subexpr(e, descend, &mut |sub| !go(sub, descend, pred))
    }
    go(e, descend, &mut pred)
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
    any_stmt(stmts, descend, |s| {
        stmt_all_exprs(s).into_iter().any(|e| any_expr_for(e, descend, &mut pred))
    })
}
