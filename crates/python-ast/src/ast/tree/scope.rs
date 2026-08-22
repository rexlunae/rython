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
    /// Names that are possibly-uninitialized where a generated closure is
    /// created (try-body, and finally-guarded handler/else bodies). rustc
    /// rejects capturing a possibly-uninitialized variable (E0381), so the
    /// hoisted declaration for these names gets a `Default::default()`
    /// initializer instead of a bare `let mut x;` (issue #78).
    pub closure_captured_uninit: HashSet<String>,
    /// `for`-target names whose value is observed after the loop (a read in
    /// a later statement that no re-binding shadows). These lower to stores
    /// into the hoisted binding, never shadowing fresh bindings (issue #80).
    pub leaked_loop_targets: HashSet<String>,
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
    // File-object methods take &mut self (reads advance the cursor);
    // csv.Writer's row methods write through it.
    "read",
    "readline",
    "readlines",
    "write",
    "writelines",
    "close",
    "writerow",
    "writerows",
];

struct Analysis<'r> {
    assigned: Vec<String>,
    /// Names assigned so far in THIS walk (sequential), as opposed to
    /// `assigned`, which is seeded with the collector's full-function list.
    /// Closure boundaries record against the sequential list so a name first
    /// assigned AFTER a try statement is not given a dummy initializer for a
    /// closure it is never captured by (Devin review on #103).
    seq_assigned: Vec<String>,
    needs_mut: HashSet<String>,
    optional: HashSet<String>,
    closure_captured_uninit: HashSet<String>,
    leaked_loop_targets: HashSet<String>,
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
        if !self.seq_assigned.contains(&name.to_string()) {
            self.seq_assigned.push(name.to_string());
        }
        if multi || self.init_of(name) != Init::No {
            self.needs_mut.insert(name.to_string());
        }
        self.state.insert(name.to_string(), Init::Yes);
    }

    /// A generated closure is about to run over `entry_state`: every name
    /// that is not definitely initialized there will be captured
    /// possibly-uninitialized, which rustc rejects (E0381). Record them so
    /// the hoist can give them a Default initializer. Only names assigned
    /// before this point in the walk (or inside the closure's own body,
    /// which the caller has walked first) can be captured — a name first
    /// assigned later in the function is not in `seq_assigned` and is left
    /// alone.
    fn record_closure_boundary(&mut self, entry_state: &HashMap<String, Init>) {
        for name in &self.seq_assigned {
            if entry_state.get(name).copied().unwrap_or(Init::No) != Init::Yes {
                self.closure_captured_uninit.insert(name.clone());
            }
        }
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
    // First pass: collect every name the body assigns, so the closure
    // boundaries in the analysis pass know the full name set (a name first
    // assigned *inside* a try body is not yet in `assigned` when the
    // boundary runs). The collector uses a no-op resolver: only the
    // `assigned` list is read back, and consulting the real resolver here
    // would poison its cycle-guard `visited` set for the analysis pass.
    let noop_resolve = |_: &crate::Call| -> Option<bool> { None };
    let mut collector = Analysis {
        assigned: Vec::new(),
        seq_assigned: Vec::new(),
        needs_mut: HashSet::new(),
        optional: HashSet::new(),
        closure_captured_uninit: HashSet::new(),
        leaked_loop_targets: HashSet::new(),
        state: HashMap::new(),
        resolve_call: &noop_resolve,
    };
    walk_stmts(body, &mut collector, false);

    let mut a = Analysis {
        assigned: collector.assigned.clone(),
        seq_assigned: Vec::new(),
        needs_mut: HashSet::new(),
        optional: HashSet::new(),
        closure_captured_uninit: HashSet::new(),
        leaked_loop_targets: HashSet::new(),
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
        closure_captured_uninit: a.closure_captured_uninit,
        leaked_loop_targets: a.leaked_loop_targets,
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

/// Whether `expr` is the real Python `super()` shape: a bare `super()` call
/// (no arguments). The receiver of `super().m(...)` — its mutation targets
/// the enclosing method's self even though the chain bottoms out in a Call.
fn is_super_call(expr: &ExprType) -> bool {
    matches!(
        expr,
        ExprType::Call(c)
            if c.args.is_empty()
                && c.keywords.is_empty()
                && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "super")
    )
}

/// Collect the names in a `for` target (`i` and `v` in `for i, v in ...`).
fn for_target_names<'a>(target: &'a ExprType, out: &mut Vec<&'a str>) {
    match target {
        ExprType::Name(n) => out.push(&n.id),
        ExprType::Tuple(t) => {
            for elt in &t.elts {
                for_target_names(elt, out);
            }
        }
        _ => {}
    }
}

/// Whether a `for` target name can be observed after the loop: a READ in a
/// later statement of the enclosing list that is not shadowed by a
/// re-binding of the name. Only then must the loop store into the hoisted
/// binding instead of using a fresh per-loop binding (issue #80).
fn for_target_referenced_outside<'a>(
    body: &'a [Statement],
    this: &'a Statement,
    name: &str,
) -> bool {
    let mut past = false;
    for s in body {
        if std::ptr::eq(s, this) {
            past = true;
            continue;
        }
        if past && statement_observes_read(s, name) {
            return true;
        }
    }
    false
}

/// Whether executing `stmt` can READ `name` in a way that observes a value
/// written before `stmt` ran — i.e. the read is not dominated by a `for`
/// target binding of `name` inside `stmt`. Plain `statement_references`
/// also counts stores and target bindings (a later `for i in ...` would
/// look like a reference and force a spurious hoist, and a hoisted
/// binding that is never actually read after the loop can end up
/// uninitialized or with the wrong type); this walker only counts reads
/// that escape.
fn statement_observes_read(stmt: &Statement, name: &str) -> bool {
    match &stmt.statement {
        StatementType::Assign(a) => crate::expr_references(&a.value, name),
        StatementType::AugAssign(a) => {
            crate::expr_references(&a.target, name) || crate::expr_references(&a.value, name)
        }
        StatementType::Expr(e) => crate::expr_references(&e.value, name),
        StatementType::Return(r) => r
            .as_ref()
            .map(|e| crate::expr_references(&e.value, name))
            .unwrap_or(false),
        StatementType::If(s) => {
            crate::expr_references(&s.test, name)
                || s.body.iter().any(|b| statement_observes_read(b, name))
                || s.orelse.iter().any(|b| statement_observes_read(b, name))
        }
        StatementType::While(s) => {
            crate::expr_references(&s.test, name)
                || s.body.iter().any(|b| statement_observes_read(b, name))
                || s.orelse.iter().any(|b| statement_observes_read(b, name))
        }
        StatementType::For(s) => {
            for_statement_observes_read(&s.target, &s.iter, &s.body, &s.orelse, name)
        }
        StatementType::AsyncFor(s) => {
            for_statement_observes_read(&s.target, &s.iter, &s.body, &s.orelse, name)
        }
        StatementType::With(s) => {
            s.items
                .iter()
                .any(|i| crate::expr_references(&i.context_expr, name))
                || s.body.iter().any(|b| statement_observes_read(b, name))
        }
        StatementType::Try(s) => {
            s.body.iter().any(|b| statement_observes_read(b, name))
                || s.handlers.iter().any(|h| {
                    h.exception_type
                        .as_ref()
                        .map(|t| crate::expr_references(t, name))
                        .unwrap_or(false)
                        || h.body.iter().any(|b| statement_observes_read(b, name))
                })
                || s.orelse.iter().any(|b| statement_observes_read(b, name))
                || s.finalbody.iter().any(|b| statement_observes_read(b, name))
        }
        _ => false,
    }
}

/// A loop whose target binds `name` shadows body reads (each iteration
/// sees the fresh value), but its iterable (evaluated before any binding)
/// and its orelse (which can run with zero iterations, leaving the earlier
/// value visible) still observe.
fn for_statement_observes_read(
    target: &ExprType,
    iter: &ExprType,
    body: &[Statement],
    orelse: &[Statement],
    name: &str,
) -> bool {
    let rebinds = {
        let mut names = Vec::new();
        for_target_names(target, &mut names);
        names.contains(&name)
    };
    crate::expr_references(iter, name)
        || orelse.iter().any(|b| statement_observes_read(b, name))
        || (!rebinds && body.iter().any(|b| statement_observes_read(b, name)))
}

/// Record stores for the `for`-target names whose values leak out of the
/// loop (observable reads in later statements). Loop-local names stay
/// unrecorded so they keep their fresh per-loop binding (and the unused-
/// index `_` lowering); only the leaked ones are hoisted, and the loop
/// lowering is told exactly which names those are so it stores into the
/// hoisted binding rather than a shadowing fresh one.
fn record_leaking_for_targets(
    target: &ExprType,
    body: &[Statement],
    this: &Statement,
    a: &mut Analysis<'_>,
) {
    let mut names = Vec::new();
    for_target_names(target, &mut names);
    for name in names {
        if for_target_referenced_outside(body, this, name) {
            // multi=true: the store runs once per iteration (possibly zero
            // times), so the binding always needs `mut` when hoisted.
            a.record_store(name, true);
            a.leaked_loop_targets.insert(name.to_string());
        }
    }
}

/// The standard call resolver for analyze_scope_with, backed by the symbol
/// table: a call whose receiver class is statically known resolves to that
/// class's own method, and the method's receiver kind (&self / &mut self)
/// decides whether the call mutates.
pub(crate) fn class_call_resolver<'a>(
    ctx: &'a crate::CodeGenContext,
    symbols: &'a crate::SymbolTableScopes,
    options: &'a crate::PythonOptions,
) -> impl Fn(&crate::Call) -> Option<bool> + 'a {
    move |call| {
        let ExprType::Attribute(attr) = call.func.as_ref() else {
            return None;
        };
        let (class, class_symbols) = crate::receiver_class(&attr.value, ctx, symbols, options)?;
        if class.method_on_mro(&attr.attr, &class_symbols).is_none() {
            return None;
        }
        Some(class.method_needs_mut_self(&attr.attr, &class_symbols, options))
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
            StatementType::Delete(targets) => {
                // `del xs[i]` mutates the container through py_pop.
                for target in targets {
                    if let ExprType::Subscript(s) = target {
                        walk_expr(&s.value, a);
                        if let ExprType::Name(name) = s.value.as_ref() {
                            a.record_mutation(&name.id);
                        }
                        if let Some(idx) = crate::SubscriptKindExpr::kind_expr(s) {
                            walk_expr(idx, a);
                        }
                    }
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
                record_leaking_for_targets(&s.target, body, stmt, a);
                walk_loop(&s.body, &s.orelse, a, multi);
            }
            StatementType::AsyncFor(s) => {
                walk_expr(&s.iter, a);
                record_leaking_for_targets(&s.target, body, stmt, a);
                walk_loop(&s.body, &s.orelse, a, multi);
            }
            StatementType::Try(s) => {
                // The try body runs inside a closure; stores there behave
                // like multi-execution stores (they mutate captured state).
                let before = a.state.clone();
                walk_stmts(&s.body, a, true);
                let after_body = a.state.clone();
                // Record the boundary AFTER walking the body: names first
                // assigned inside the try body are in `seq_assigned` now
                // (their stores capture the hoisted binding, which is
                // possibly-uninitialized when the closure is created —
                // E0381), while names assigned later in the function are
                // not, so they get no dummy initializer.
                a.record_closure_boundary(&before);
                // Handlers may run with the body only partially executed.
                let handler_entry =
                    merge_states(vec![before, after_body.clone()]);
                // With a finally clause the handler and else bodies also run
                // in closures (so a return/raise still executes finally):
                // their entry state may still be uninitialized for names
                // first assigned inside the try statement.
                if !s.finalbody.is_empty() {
                    a.record_closure_boundary(&handler_entry);
                    a.record_closure_boundary(&after_body);
                }
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
    // csv.writer(f) holds &mut f for the writer's lifetime.
    "writer",
];

fn walk_call(call: &crate::Call, a: &mut Analysis<'_>) {
    if let ExprType::Attribute(attr) = call.func.as_ref() {
        // The module-prefixed spelling mutates its first ARGUMENT, not the
        // receiver: `heapq.heappush(h, x)` needs `h` mutable, mirroring
        // the bare-function branch below.
        if let ExprType::Name(m) = attr.value.as_ref() {
            if matches!(m.id.as_str(), "heapq" | "csv")
                && FIRST_ARG_MUTATORS.contains(&attr.attr.as_str())
            {
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
            } else if is_super_call(&attr.value) {
                // `super().m(...)` mutates the ENCLOSING method's self:
                // the receiver chain bottoms out in a `super()` call, not a
                // name, but the mutation is on the method's own object
                // (super().__init__ writes the base's fields).
                a.record_mutation("self");
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
            // The walrus target binds in the ENCLOSING scope (Python
            // semantics): `if (x := f()) is not None:` reads x in the body.
            // Record it as a store so it hoists to the function's
            // declarations and the walrus renders as a store into the
            // hoisted binding (issue #137 cluster — urllib3's http2
            // `if data_to_send := conn.data_to_send():`).
            if let ExprType::Name(n) = e.left.as_ref() {
                a.record_store(&n.id, false);
            }
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
