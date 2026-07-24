//! Scope analysis for variable hoisting.
//!
//! Python variables are function-scoped, so assigned names are hoisted to
//! declarations at the top of the generated scope and assignments lower to
//! plain stores (see `Assign`). This module decides which of those
//! declarations actually need `mut`, mirroring rustc's own
//! definite-initialization rules so the generated code carries neither
//! `unused_mut` warnings nor missing-`mut` errors:
//!
//! - A store into a variable that may already hold a value (reassignment,
//!   assignment on a path where it was maybe-assigned, any assignment inside
//!   a loop or try-closure) needs `mut`.
//! - The first store into a definitely-uninitialized variable is
//!   initialization and does not.
//! - Augmented assignment, storing through the variable (`x[i] = v`,
//!   `x.f = v`), and calling a known-mutating Python method on it
//!   (`x.append(...)`) need `mut`.

use std::collections::{HashMap, HashSet};

use crate::{ExprType, Statement, StatementType};

/// The binding facts for one generated scope.
pub struct ScopeBindings {
    /// Names bound by `Assign` statements, in first-assignment order.
    pub assigned: Vec<String>,
    /// The subset of names (assigned names and parameters) that need a
    /// mutable binding.
    pub needs_mut: HashSet<String>,
    /// Names assigned `None` on some path: they hold an Option, and every
    /// non-None store into them wraps in `Some`.
    pub optional: HashSet<String>,
}

/// How definitely a variable holds a value at a program point.
#[derive(Clone, Copy, PartialEq)]
enum Init {
    No,
    Maybe,
    Yes,
}

/// Python methods that mutate their receiver; calling one on a name means
/// the binding must be mutable. An unlisted mutating method surfaces as a
/// missing-`mut` compile error in the generated code (loud, not silent).
pub(crate) const MUTATING_METHODS: &[&str] = &[
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "popitem",
    "clear",
    "sort",
    "reverse",
    "update",
    "add",
    "discard",
    "setdefault",
    "intersection_update",
    "difference_update",
    "symmetric_difference_update",
    "push",
];

struct Analysis<'r> {
    assigned: Vec<String>,
    needs_mut: HashSet<String>,
    optional: HashSet<String>,
    state: HashMap<String, Init>,
    /// Classifies a method call's effect on its receiver when the
    /// receiver's class is statically known: Some(needs_mut) resolves the
    /// question authoritatively (a user method shadowing a builtin mutator
    /// name may be read-only); None means unknown, falling back to the
    /// syntactic MUTATING_METHODS list.
    resolve_call: &'r dyn Fn(&crate::Call) -> Option<bool>,
}

impl Analysis<'_> {
    fn init_of(&self, name: &str) -> Init {
        self.state.get(name).copied().unwrap_or(Init::No)
    }

    /// Record a store. `multi` is true when the store may execute more than
    /// once (loop body) or runs inside the try-block closure.
    fn record_store(&mut self, name: &str, multi: bool) {
        if !self.assigned.contains(&name.to_string()) {
            self.assigned.push(name.to_string());
        }
        if multi || self.init_of(name) != Init::No {
            self.needs_mut.insert(name.to_string());
        }
        self.state.insert(name.to_string(), Init::Yes);
    }

    /// The variable is mutated in place (aug-assign, store-through, or a
    /// mutating method call).
    fn record_mutation(&mut self, name: &str) {
        self.needs_mut.insert(name.to_string());
    }
}

/// Merge the post-states of alternative branches.
fn merge_states(
    branches: Vec<HashMap<String, Init>>,
) -> HashMap<String, Init> {
    let mut merged: HashMap<String, Init> = HashMap::new();
    let keys: HashSet<String> = branches
        .iter()
        .flat_map(|b| b.keys().cloned())
        .collect();
    for key in keys {
        let states: Vec<Init> = branches
            .iter()
            .map(|b| b.get(&key).copied().unwrap_or(Init::No))
            .collect();
        let all_yes = states.iter().all(|s| *s == Init::Yes);
        let all_no = states.iter().all(|s| *s == Init::No);
        let v = if all_yes {
            Init::Yes
        } else if all_no {
            Init::No
        } else {
            Init::Maybe
        };
        merged.insert(key, v);
    }
    merged
}

/// Analyze a statement list (one generated scope). `initialized` names —
/// the parameters — hold values at entry, so any store into them needs a
/// mutable rebinding.
pub fn analyze_scope(body: &[Statement], initialized: &[String]) -> ScopeBindings {
    analyze_scope_with(body, initialized, &|_| None)
}

/// analyze_scope with a call resolver: when the resolver classifies a
/// method call (receiver class statically known), its answer is
/// authoritative for whether the call mutates the receiver chain's base
/// binding; unresolved calls fall back to the syntactic method-name list.
pub(crate) fn analyze_scope_with(
    body: &[Statement],
    initialized: &[String],
    resolve_call: &dyn Fn(&crate::Call) -> Option<bool>,
) -> ScopeBindings {
    let mut a = Analysis {
        assigned: Vec::new(),
        needs_mut: HashSet::new(),
        optional: HashSet::new(),
        state: initialized
            .iter()
            .map(|n| (n.clone(), Init::Yes))
            .collect(),
        resolve_call,
    };
    walk_stmts(body, &mut a, false);
    // Parameters are tracked for needs_mut but are not hoisted declarations.
    let assigned = a
        .assigned
        .into_iter()
        .filter(|n| !initialized.contains(n))
        .collect();
    ScopeBindings {
        assigned,
        needs_mut: a.needs_mut,
        optional: a.optional,
    }
}

fn record_target(target: &ExprType, a: &mut Analysis<'_>, multi: bool) {
    match target {
        ExprType::Name(name) => a.record_store(&name.id, multi),
        ExprType::Tuple(tuple) => {
            for elt in &tuple.elts {
                // Destructuring assignment; conservatively mutable.
                if let ExprType::Name(name) = elt {
                    a.record_store(&name.id, multi);
                    a.record_mutation(&name.id);
                } else {
                    record_target(elt, a, multi);
                }
            }
        }
        // Stores through the base name: `x[i] = v`, `x.f = v`, and nested
        // chains like `grid[i][j] = v` all mutate the chain's base binding.
        ExprType::Subscript(sub) => {
            if let Some(name) = chain_base_name(&sub.value) {
                a.record_mutation(name);
            }
            walk_subscript_kind(&sub.kind, a);
        }
        ExprType::Attribute(attr) => {
            if let Some(name) = chain_base_name(&attr.value) {
                a.record_mutation(name);
            }
        }
        _ => {}
    }
}

/// The name at the base of a subscript/attribute chain (`grid` in
/// `grid[i][j]` or `obj.rows[i]`), if the chain bottoms out in one.
pub(crate) fn chain_base_name(expr: &ExprType) -> Option<&str> {
    match expr {
        ExprType::Name(name) => Some(&name.id),
        ExprType::Subscript(sub) => chain_base_name(&sub.value),
        ExprType::Attribute(attr) => chain_base_name(&attr.value),
        _ => None,
    }
}

/// The standard call resolver for analyze_scope_with, backed by the symbol
/// table: a call whose receiver class is statically known resolves to that
/// class's own method, and the method's receiver kind (&self / &mut self)
/// decides whether the call mutates.
pub(crate) fn class_call_resolver<'a>(
    ctx: &'a crate::CodeGenContext,
    symbols: &'a crate::SymbolTableScopes,
) -> impl Fn(&crate::Call) -> Option<bool> + 'a {
    move |call| {
        let ExprType::Attribute(attr) = call.func.as_ref() else {
            return None;
        };
        let class = crate::receiver_class(&attr.value, ctx, symbols)?;
        if !class.methods().any(|m| m.name == attr.attr) {
            return None;
        }
        Some(class.method_needs_mut_self(&attr.attr, symbols))
    }
}

fn walk_stmts(body: &[Statement], a: &mut Analysis<'_>, multi: bool) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Assign(assign) => {
                walk_expr(&assign.value, a);
                let value_is_none = crate::is_none_expr(&assign.value);
                for target in &assign.targets {
                    record_target(target, a, multi);
                    if value_is_none {
                        if let ExprType::Name(name) = target {
                            a.optional.insert(name.id.clone());
                        }
                    }
                }
            }
            StatementType::AugAssign(aug) => {
                walk_expr(&aug.value, a);
                if let ExprType::Name(name) = &aug.target {
                    // Reads and writes an existing value: always mutable.
                    a.record_store(&name.id, multi);
                    a.record_mutation(&name.id);
                } else {
                    record_target(&aug.target, a, multi);
                }
            }
            StatementType::Expr(e) => walk_expr(&e.value, a),
            StatementType::Call(call) => walk_call(call, a),
            StatementType::Return(Some(e)) => walk_expr(&e.value, a),
            StatementType::Assert { test, msg } => {
                walk_expr(test, a);
                if let Some(m) = msg {
                    walk_expr(m, a);
                }
            }
            StatementType::Raise(raise) => {
                if let Some(exc) = &raise.exc {
                    walk_expr(exc, a);
                }
                if let Some(cause) = &raise.cause {
                    walk_expr(cause, a);
                }
            }
            StatementType::If(s) => {
                walk_expr(&s.test, a);
                let before = a.state.clone();
                walk_stmts(&s.body, a, multi);
                let after_body = std::mem::replace(&mut a.state, before);
                walk_stmts(&s.orelse, a, multi);
                let after_else = std::mem::replace(&mut a.state, HashMap::new());
                a.state = merge_states(vec![after_body, after_else]);
            }
            StatementType::While(s) => {
                walk_expr(&s.test, a);
                walk_loop(&s.body, &s.orelse, a, multi);
            }
            StatementType::For(s) => {
                walk_expr(&s.iter, a);
                walk_loop(&s.body, &s.orelse, a, multi);
            }
            StatementType::AsyncFor(s) => {
                walk_expr(&s.iter, a);
                walk_loop(&s.body, &s.orelse, a, multi);
            }
            StatementType::Try(s) => {
                // The try body runs inside a closure; stores there behave
                // like multi-execution stores (they mutate captured state).
                let before = a.state.clone();
                walk_stmts(&s.body, a, true);
                let after_body = a.state.clone();
                // Handlers may run with the body only partially executed.
                let handler_entry =
                    merge_states(vec![before, after_body.clone()]);
                let mut exits = Vec::new();
                for handler in &s.handlers {
                    a.state = handler_entry.clone();
                    walk_stmts(&handler.body, a, multi);
                    exits.push(std::mem::replace(&mut a.state, HashMap::new()));
                }
                // The else path continues from the completed body.
                a.state = after_body;
                walk_stmts(&s.orelse, a, multi);
                exits.push(std::mem::replace(&mut a.state, HashMap::new()));
                a.state = merge_states(exits);
                walk_stmts(&s.finalbody, a, multi);
            }
            StatementType::With(s) => {
                for item in &s.items {
                    walk_expr(&item.context_expr, a);
                }
                walk_stmts(&s.body, a, multi);
            }
            StatementType::AsyncWith(s) => {
                for item in &s.items {
                    walk_expr(&item.context_expr, a);
                }
                walk_stmts(&s.body, a, multi);
            }
            // Nested definitions are their own scopes.
            _ => {}
        }
    }
}

/// A loop body may execute any number of times: analyze it as a branch that
/// may or may not have run, with every store marked multi-execution.
fn walk_loop(
    body: &[Statement],
    orelse: &[Statement],
    a: &mut Analysis<'_>,
    outer_multi: bool,
) {
    let before = a.state.clone();
    walk_stmts(body, a, true);
    let after_body = std::mem::replace(&mut a.state, HashMap::new());
    a.state = merge_states(vec![before, after_body]);
    walk_stmts(orelse, a, outer_multi);
}

/// Free functions that mutate their first argument in place: the heapq
/// surface treats a plain list as a heap, so `heappush(h, x)` mutates `h`
/// like a method call would.
const FIRST_ARG_MUTATORS: &[&str] = &[
    "heappush",
    "heappop",
    "heapify",
    "heappushpop",
    "heapreplace",
];

fn walk_call(call: &crate::Call, a: &mut Analysis<'_>) {
    if let ExprType::Attribute(attr) = call.func.as_ref() {
        // The module-prefixed spelling mutates its first ARGUMENT, not the
        // receiver: `heapq.heappush(h, x)` needs `h` mutable, mirroring
        // the bare-function branch below.
        if let ExprType::Name(m) = attr.value.as_ref() {
            if m.id == "heapq" && FIRST_ARG_MUTATORS.contains(&attr.attr.as_str()) {
                if let Some(first) = call.args.first() {
                    if let Some(name) = chain_base_name(first) {
                        a.record_mutation(name);
                    }
                }
            }
        }
        // A mutating method mutates the base binding of the whole receiver
        // chain: `self.items.append(x)` mutates `self`, `rows[i].push(x)`
        // mutates `rows`. When the receiver's class is statically known,
        // the resolver's verdict is authoritative — a user method may
        // shadow a builtin mutator name yet be read-only, or mutate under
        // a name the syntactic list doesn't know.
        let mutates = match (a.resolve_call)(call) {
            Some(verdict) => verdict,
            None => MUTATING_METHODS.contains(&attr.attr.as_str()),
        };
        if mutates {
            if let Some(name) = chain_base_name(&attr.value) {
                a.record_mutation(name);
            }
        }
        walk_expr(&attr.value, a);
    } else {
        // Free functions that mutate their first argument in place; see
        // FIRST_ARG_MUTATORS.
        if let ExprType::Name(n) = call.func.as_ref() {
            if FIRST_ARG_MUTATORS.contains(&n.id.as_str()) {
                if let Some(first) = call.args.first() {
                    if let Some(name) = chain_base_name(first) {
                        a.record_mutation(name);
                    }
                }
            }
        }
        walk_expr(&call.func, a);
    }
    for arg in &call.args {
        walk_expr(arg, a);
    }
    // Keyword-argument values carry mutations too: `foo(x=c.bump())`
    // mutates `c` just as surely as a positional argument would.
    for kw in &call.keywords {
        walk_expr(&kw.value, a);
    }
}

fn walk_subscript_kind(kind: &crate::SubscriptKind, a: &mut Analysis<'_>) {
    match kind {
        crate::SubscriptKind::Index(i) => walk_expr(i, a),
        crate::SubscriptKind::Slice { lower, upper, step } => {
            for bound in [lower, upper, step].into_iter().flatten() {
                walk_expr(bound, a);
            }
        }
    }
}

fn walk_expr(expr: &ExprType, a: &mut Analysis<'_>) {
    match expr {
        ExprType::Call(call) => walk_call(call, a),
        ExprType::BinOp(op) => {
            walk_expr(&op.left, a);
            walk_expr(&op.right, a);
        }
        ExprType::BoolOp(op) => {
            for v in &op.values {
                walk_expr(v, a);
            }
        }
        ExprType::UnaryOp(op) => walk_expr(&op.operand, a),
        ExprType::Compare(cmp) => {
            walk_expr(&cmp.left, a);
            for c in &cmp.comparators {
                walk_expr(c, a);
            }
        }
        ExprType::IfExp(e) => {
            walk_expr(&e.test, a);
            walk_expr(&e.body, a);
            walk_expr(&e.orelse, a);
        }
        ExprType::NamedExpr(e) => {
            walk_expr(&e.left, a);
            walk_expr(&e.right, a);
        }
        ExprType::Dict(d) => {
            for k in d.keys.iter().flatten() {
                walk_expr(k, a);
            }
            for v in &d.values {
                walk_expr(v, a);
            }
        }
        ExprType::Set(s) => {
            for e in &s.elts {
                walk_expr(e, a);
            }
        }
        ExprType::List(elts) => {
            for e in elts {
                walk_expr(e, a);
            }
        }
        ExprType::Tuple(t) => {
            for e in &t.elts {
                walk_expr(e, a);
            }
        }
        ExprType::Attribute(attr) => walk_expr(&attr.value, a),
        ExprType::Subscript(sub) => {
            walk_expr(&sub.value, a);
            walk_subscript_kind(&sub.kind, a);
        }
        ExprType::Starred(s) => walk_expr(&s.value, a),
        ExprType::Await(e) => walk_expr(&e.value, a),
        ExprType::Yield(y) => {
            if let Some(v) = &y.value {
                walk_expr(v, a);
            }
        }
        ExprType::YieldFrom(y) => walk_expr(&y.value, a),
        ExprType::FormattedValue(f) => walk_expr(&f.value, a),
        ExprType::JoinedStr(j) => {
            for v in &j.values {
                walk_expr(v, a);
            }
        }
        // Comprehensions and lambdas are their own scopes; leaves carry no
        // mutation.
        _ => {}
    }
}
