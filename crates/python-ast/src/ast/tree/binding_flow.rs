//! Definite bindings at local reads, separate from Rust mutability/hoisting.
//!
//! A returned/raised branch cannot make a later read unbound. Keep exits
//! apart until their destination (a loop, handler, or finally), intersecting
//! bindings only on paths reaching that point. All syntax traversal belongs
//! to `visit`; the matches here choose how its children exchange flow facts.

use std::collections::{HashMap, HashSet};

use super::visit::{self, Bindings, Descend, Flow};
use crate::{ExprType, Statement, StatementType};

type Names = HashSet<String>;

#[derive(Clone, Copy)]
enum Exit {
    Next,
    Return,
    Raise,
    Break,
    Continue,
}

const EXITS: [Exit; 5] = [
    Exit::Next,
    Exit::Return,
    Exit::Raise,
    Exit::Break,
    Exit::Continue,
];

/// None is unreachable; Some(empty) is reachable with no definite bindings.
#[derive(Default)]
struct Paths([Option<Names>; 5]);

fn join(into: &mut Option<Names>, other: Option<Names>) {
    if let Some(other) = other {
        if let Some(current) = into {
            current.retain(|name| other.contains(name));
        } else {
            *into = Some(other);
        }
    }
}

impl Paths {
    fn one(exit: Exit, bound: Names) -> Self {
        let mut paths = Self::default();
        paths.0[exit as usize] = Some(bound);
        paths
    }

    fn take(&mut self, exit: Exit) -> Option<Names> {
        self.0[exit as usize].take()
    }

    fn merge(&mut self, mut other: Self) {
        for exit in EXITS {
            join(&mut self.0[exit as usize], other.take(exit));
        }
    }
}

struct Analysis<'a> {
    watched: &'a Names,
    reads: HashMap<String, usize>,
}

impl Analysis<'_> {
    fn bind(&self, bound: &mut Names, names: impl IntoIterator<Item = String>) {
        bound.extend(names.into_iter().filter(|name| self.watched.contains(name)));
    }

    fn expr(&mut self, expr: &ExprType, bound: &mut Names, line: usize) {
        if let ExprType::Name(name) = expr
            && self.watched.contains(&name.id)
            && !bound.contains(&name.id)
        {
            self.reads
                .entry(name.id.clone())
                .and_modify(|old| *old = (*old).min(line))
                .or_insert(line);
        }
        if let Some((elements, generators)) = visit::comprehension_parts(expr) {
            if let Some(first) = generators.first() {
                // Only the first iterable runs in the enclosing scope.
                self.expr(&first.iter, bound, line);
            }
            let mut inner = bound.clone();
            // All iteration targets are locals of the comprehension,
            // including a later generator's not-yet-initialized target.
            for generator in generators {
                for name in visit::target_names(&generator.target) {
                    inner.remove(name);
                }
            }
            for (index, generator) in generators.iter().enumerate() {
                if index > 0 {
                    self.expr(&generator.iter, &mut inner, line);
                }
                self.target(&generator.target, &mut inner, line, false);
                for filter in &generator.ifs {
                    self.expr(filter, &mut inner, line);
                }
            }
            for element in elements {
                self.expr(element, &mut inner, line);
            }
            // Targets never leak out. A walrus binds the enclosing local
            // and is visible to subsequent inner expressions, but no
            // iteration is guaranteed, so it adds no definite outer fact.
            return;
        }
        let children = visit::subexprs_for(expr, Descend::OwnScope);
        match expr {
            ExprType::NamedExpr(_) => {
                // The visitor exposes target, value; Python evaluates the
                // value before binding the target (which is not a load).
                self.expr(children[1], bound, line);
                self.bind(
                    bound,
                    visit::target_names(children[0])
                        .into_iter()
                        .map(str::to_owned),
                );
            }
            ExprType::IfExp(_) => {
                self.expr(children[0], bound, line);
                let mut yes = bound.clone();
                self.expr(children[1], &mut yes, line);
                self.expr(children[2], bound, line);
                bound.retain(|name| yes.contains(name));
            }
            ExprType::BoolOp(_) | ExprType::Compare(_) => {
                // Later operands/comparators may short-circuit. They see
                // preceding stores, but those stores need not survive the
                // whole expression. The first comparison pair always runs.
                let required = if matches!(expr, ExprType::Compare(_)) {
                    2
                } else {
                    1
                };
                let mut exits = None;
                for (i, child) in children.into_iter().enumerate() {
                    self.expr(child, bound, line);
                    if i + 1 >= required {
                        join(&mut exits, Some(bound.clone()));
                    }
                }
                if let Some(common) = exits {
                    *bound = common;
                }
            }
            ExprType::Dict(_) => {
                // The visitor enumerates keys then values, not dict
                // evaluation order. Check each against the incoming facts
                // instead of allowing a later key's walrus to hide an
                // earlier value's unbound read. All entries do run.
                let before = bound.clone();
                for child in children {
                    let mut entry = before.clone();
                    self.expr(child, &mut entry, line);
                    bound.extend(entry);
                }
            }
            _ => {
                for child in children {
                    self.expr(child, bound, line);
                }
            }
        }
    }

    fn target(&mut self, expr: &ExprType, bound: &mut Names, line: usize, load: bool) {
        match expr {
            ExprType::Name(_) => {
                if load {
                    self.expr(expr, bound, line);
                }
                self.bind(
                    bound,
                    visit::target_names(expr).into_iter().map(str::to_owned),
                );
            }
            ExprType::Tuple(_) | ExprType::List(_) | ExprType::Starred(_) => {
                for child in visit::subexprs_for(expr, Descend::OwnScope) {
                    self.target(child, bound, line, load);
                }
            }
            // Assigning an attribute/item evaluates its receiver/index.
            _ => self.expr(expr, bound, line),
        }
    }

    fn block(&mut self, body: &[Statement], entry: Names) -> Paths {
        let mut exits = Paths::one(Exit::Next, entry);
        visit::walk_stmts(body, Descend::OwnScope, &mut |stmt| {
            let Some(bound) = exits.take(Exit::Next) else {
                return Flow::Stop;
            };
            exits.merge(self.stmt(stmt, bound));
            // Visit children below through stmt_bodies_for with their
            // branch-specific facts. A nested def/class body is its own
            // scope and deliberately excluded by OwnScope.
            Flow::Skip
        });
        exits
    }

    fn stmt(&mut self, stmt: &Statement, mut bound: Names) -> Paths {
        let line = stmt.lineno.unwrap_or(0);
        let before = bound.clone();
        let exprs = visit::stmt_exprs(stmt);
        for expr in &exprs {
            self.expr(expr, &mut bound, line);
        }
        let bodies = visit::stmt_bodies_for(stmt, Descend::OwnScope);
        let mut exits = match &stmt.statement {
            StatementType::If(_) => {
                let mut arms = self.block(bodies[0], bound.clone());
                arms.merge(self.block(bodies[1], bound));
                arms
            }
            StatementType::For(_) | StatementType::AsyncFor(_) | StatementType::While(_) => {
                let mut head = bound.clone();
                let mut all = Paths::default();
                loop {
                    let mut iteration = head.clone();
                    if matches!(stmt.statement, StatementType::While(_)) {
                        for expr in &exprs {
                            self.expr(expr, &mut iteration, line);
                        }
                    }
                    for target in visit::stmt_targets(stmt) {
                        self.target(target, &mut iteration, line, false);
                    }
                    let mut body = self.block(bodies[0], iteration);
                    let mut back = body.take(Exit::Next);
                    join(&mut back, body.take(Exit::Continue));
                    join(&mut all.0[Exit::Next as usize], body.take(Exit::Break));
                    all.merge(body);
                    let mut next_head = Some(bound.clone()); // zero iterations
                    join(&mut next_head, back);
                    let next_head = next_head.unwrap();
                    if next_head == head {
                        break;
                    }
                    head = next_head;
                }
                // Exhaustion enters else; breaks bypass it. Loop-local
                // break/continue paths never become function exits.
                if matches!(stmt.statement, StatementType::While(_)) {
                    for expr in &exprs {
                        self.expr(expr, &mut head, line);
                    }
                }
                all.merge(self.block(bodies[1], head));
                all
            }
            StatementType::Try(tried) => {
                let mut body = self.block(bodies[0], bound);
                let handler_entry = body.0[Exit::Raise as usize].clone();
                let mut paths = Paths::default();
                if let Some(normal) = body.take(Exit::Next) {
                    paths.merge(self.block(bodies[bodies.len() - 2], normal));
                }
                // A bare except handles every exception. Other handlers
                // may leave an unmatched exception to propagate outward.
                if tried.handlers.iter().any(|h| h.exception_type.is_none()) {
                    body.take(Exit::Raise);
                }
                paths.merge(body);
                if let Some(entry) = handler_entry {
                    for (handler, body) in tried.handlers.iter().zip(&bodies[1..]) {
                        let mut entry = entry.clone();
                        if let Some(expr) = &handler.exception_type {
                            self.expr(expr, &mut entry, handler.lineno.unwrap_or(line));
                        }
                        if let Some(name) = &handler.name {
                            self.bind(&mut entry, [name.clone()]);
                        }
                        let mut handler_paths = self.block(body, entry);
                        // CPython clears the target on every handler exit,
                        // including returns/raises that execute finally.
                        for names in handler_paths.0.iter_mut().flatten() {
                            if let Some(name) = &handler.name {
                                names.remove(name);
                            }
                        }
                        paths.merge(handler_paths);
                    }
                }
                let mut result = Paths::default();
                for exit in EXITS {
                    if let Some(entry) = paths.take(exit) {
                        let mut finally = self.block(bodies[bodies.len() - 1], entry);
                        // A normally completed finally resumes the old
                        // exit; an abrupt finally replaces it.
                        join(&mut result.0[exit as usize], finally.take(Exit::Next));
                        result.merge(finally);
                    }
                }
                result
            }
            StatementType::With(_) | StatementType::AsyncWith(_) => {
                for target in visit::stmt_targets(stmt) {
                    self.target(target, &mut bound, line, false);
                }
                let mut paths = self.block(bodies[0], bound);
                if let StatementType::With(with) = &stmt.statement
                    && with.items.len() == 1
                    && with.items[0].optional_vars.is_none()
                    && super::with_stmt::assert_raises_cm_name(&with.items[0].context_expr)
                        .is_some()
                {
                    // assertRaises can suppress a body exception, but
                    // never a return/break/continue.
                    let caught = paths.0[Exit::Raise as usize].clone();
                    join(&mut paths.0[Exit::Next as usize], caught);
                }
                paths
            }
            StatementType::Return(_) => Paths::one(Exit::Return, bound),
            StatementType::Raise(_) => Paths::one(Exit::Raise, bound),
            StatementType::Break => Paths::one(Exit::Break, bound),
            StatementType::Continue => Paths::one(Exit::Continue, bound),
            _ => {
                let load = matches!(stmt.statement, StatementType::AugAssign(_));
                for target in visit::stmt_targets(stmt) {
                    self.target(target, &mut bound, line, load);
                }
                if let StatementType::Delete(targets) = &stmt.statement {
                    for target in targets {
                        for name in visit::target_names(target) {
                            bound.remove(name);
                        }
                    }
                }
                if matches!(
                    stmt.statement,
                    StatementType::Import(_)
                        | StatementType::ImportFrom(_)
                        | StatementType::FunctionDef(_)
                        | StatementType::AsyncFunctionDef(_)
                        | StatementType::ClassDef(_)
                ) {
                    let conditional = visit::conditional_walrus_names(stmt);
                    self.bind(
                        &mut bound,
                        visit::stmt_bound_names(stmt, Bindings::Scope)
                            .into_iter()
                            .filter(|name| !conditional.contains(name)),
                    );
                }
                let mut paths = Paths::one(Exit::Next, bound.clone());
                // Inspect any new body-carrying form the visitor learns,
                // without assuming it binds names on every fallthrough.
                for body in bodies {
                    paths.merge(self.block(body, bound.clone()));
                }
                paths
            }
        };
        if !exprs.is_empty() || !visit::stmt_targets(stmt).is_empty() {
            // Evaluation may raise before a store completes. Handlers and
            // finally must not assume that statement's binding happened.
            join(&mut exits.0[Exit::Raise as usize], Some(before));
        }
        exits
    }
}

pub(crate) fn first_unbound_read(
    body: &[Statement],
    initialized: &[String],
    watched: &Names,
) -> Option<(String, usize)> {
    if watched.is_empty() {
        return None;
    }
    let mut analysis = Analysis {
        watched,
        reads: HashMap::new(),
    };
    let initialized = initialized
        .iter()
        .filter(|name| watched.contains(*name))
        .cloned()
        .collect();
    analysis.block(body, initialized);
    analysis
        .reads
        .into_iter()
        .min_by(|(an, al), (bn, bl)| (al, an).cmp(&(bl, bn)))
}
