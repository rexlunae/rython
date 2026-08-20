//! Conversion-time aliasing guard — issue #79's cheap intermediate step.
//!
//! rython models containers as VALUES: `b = a` copies the container, and
//! passing one to a function copies it at the call site (or clones it on
//! reuse). CPython binds a REFERENCE, so a mutation through any alias is
//! visible through all of them. The faithful fix (Rc<RefCell<...>> value
//! semantics) is a project-level decision; until then the two shapes that
//! would silently diverge are detected here and rejected loudly at
//! conversion time, naming the Python line — never silently producing a
//! different answer, and never leaving the report to rustc's move checker
//! (whose errors point at Rust concepts, not the Python source).

use std::collections::{HashMap, HashSet};

use crate::ast::tree::StatementType;
use crate::{ExprType, FunctionDef, Statement, SymbolTableNode, SymbolTableScopes, TypeInfo};

/// Methods that mutate their receiver in place, for the container types
/// rython models (list, dict, deque, set, ...).
const MUTATING_METHODS: &[&str] = &[
    "append", "appendleft", "extend", "extendleft", "insert", "pop", "popleft", "remove",
    "clear", "sort", "reverse", "rotate", "update", "setdefault", "popitem", "add",
    "discard", "intersection_update", "difference_update", "symmetric_difference_update",
];

/// The root name of a mutation target: `a`, `a[i]`, `a[i][j]`, `a.k`.
fn root_name_of(target: &ExprType) -> Option<&str> {
    match target {
        ExprType::Name(n) => Some(&n.id),
        ExprType::Subscript(s) => root_name_of(&s.value),
        ExprType::Attribute(a) => root_name_of(&a.value),
        _ => None,
    }
}

/// Run the aliasing guard over one function (or module) body.
///
/// `symbols` resolves callee names to their `FunctionDef`s for the
/// passing-a-container-to-a-mutating-function shape; `name_types` and
/// `use_counts` come from the same analysis that drives clone-on-reuse.
pub fn check_aliasing(
    body: &[Statement],
    symbols: &SymbolTableScopes,
    name_types: &HashMap<String, TypeInfo>,
    use_counts: &HashMap<String, usize>,
) -> Result<(), String> {
    let mut guard = AliasingGuard {
        use_counts,
        symbols,
        containers: name_types
            .iter()
            .filter(|(_, t)| matches!(t, TypeInfo::Vec(_) | TypeInfo::Dict(_, _)))
            .map(|(n, _)| n.clone())
            .collect(),
        aliases: Vec::new(),
        mutated: HashSet::new(),
        alias_calls: Vec::new(),
    };
    guard.walk(body);

    // Shape 1: `b = a` on a container where either name is later mutated.
    // CPython shows the mutation through both names; rython's copy would
    // not.
    for (target, source, lineno) in &guard.aliases {
        if guard.mutated.contains(target) || guard.mutated.contains(source) {
            return Err(alias_error(
                &format!(
                    "`{target} = {source}` shares one container between two names, and the \
                     container is mutated afterwards; rython copies containers by value, so \
                     the mutation would not be visible through `{target}`"
                ),
                *lineno,
            ));
        }
    }

    // Shape 2: a container name passed to a function that mutates the
    // parameter, then used again afterwards. rython clones at the call
    // site, so the mutation would not be visible through the original
    // name.
    for (name, lineno) in &guard.alias_calls {
        return Err(alias_error(
            &format!(
                "`{name}` (a container) is passed to a function that mutates it, and read \
                 again afterwards; rython copies containers at the call site, so the \
                 mutation would not be visible through `{name}`"
            ),
            *lineno,
        ));
    }

    Ok(())
}

fn alias_error(what: &str, lineno: Option<usize>) -> String {
    match lineno {
        Some(line) => format!(
            "{what} (line {line}). Python containers are references (issue #79); \
             restructure the code — mutate through a single name, or pass the \
             container and return the result"
        ),
        None => format!(
            "{what}. Python containers are references (issue #79); restructure the code — \
             mutate through a single name, or pass the container and return the result"
        ),
    }
}

struct AliasingGuard<'a> {
    use_counts: &'a HashMap<String, usize>,
    symbols: &'a SymbolTableScopes,
    /// Names known to hold mutable containers: from `name_types`, plus the
    /// source and target of every alias (an alias target's own type is not
    /// inferred — `b = a` has no syntactic type — but it IS a container).
    containers: HashSet<String>,
    aliases: Vec<(String, String, Option<usize>)>,
    mutated: HashSet<String>,
    alias_calls: Vec<(String, Option<usize>)>,
}

impl<'a> AliasingGuard<'a> {
    fn walk(&mut self, body: &[Statement]) {
        for stmt in body {
            let lineno = stmt.lineno;
            match &stmt.statement {
                StatementType::Assign(a) => self.visit_assign(a, lineno),
                StatementType::AugAssign(a) => {
                    // `a += v` / `a[i] += v` mutate the container.
                    if let Some(name) = root_name_of(&a.target) {
                        self.mutated.insert(name.to_string());
                    }
                    self.visit_expr(&a.value);
                }
                StatementType::Call(c) => {
                    self.visit_call(c, lineno);
                }
                StatementType::Expr(e) => self.visit_expr(&e.value),
                StatementType::If(i) => {
                    self.walk(&i.body);
                    self.walk(&i.orelse);
                }
                StatementType::For(f) => {
                    self.walk(&f.body);
                    self.walk(&f.orelse);
                }
                StatementType::AsyncFor(f) => {
                    self.walk(&f.body);
                    self.walk(&f.orelse);
                }
                StatementType::While(w) => {
                    self.walk(&w.body);
                    self.walk(&w.orelse);
                }
                StatementType::Try(t) => {
                    self.walk(&t.body);
                    for handler in &t.handlers {
                        self.walk(&handler.body);
                    }
                    self.walk(&t.orelse);
                    self.walk(&t.finalbody);
                }
                StatementType::With(w) => self.walk(&w.body),
                StatementType::AsyncWith(w) => self.walk(&w.body),
                StatementType::Return(Some(e)) => self.visit_expr(&e.value),
                StatementType::Raise(r) => {
                    if let Some(exc) = &r.exc {
                        self.visit_expr(exc);
                    }
                    if let Some(cause) = &r.cause {
                        self.visit_expr(cause);
                    }
                }
                StatementType::Global(_) => {}
                StatementType::Assert { test, msg } => {
                    self.visit_expr(test);
                    if let Some(m) = msg {
                        self.visit_expr(m);
                    }
                }
                // Nested functions/classes are separate scopes: their own
                // compilation runs the guard again.
                StatementType::FunctionDef(_)
                | StatementType::AsyncFunctionDef(_)
                | StatementType::ClassDef(_)
                | StatementType::Import(_)
                | StatementType::ImportFrom(_)
                | StatementType::Pass
                | StatementType::Break
                | StatementType::Continue
                | StatementType::Return(None)
                | StatementType::Unimplemented(_) => {}
            }
        }
    }

    fn visit_assign(&mut self, a: &crate::Assign, lineno: Option<usize>) {
        // Aliasing: `b = a` (and chained `a = b = c`) where the source is a
        // container-typed name.
        if let ExprType::Name(source) = &a.value {
            if self.containers.contains(&source.id) {
                for target in &a.targets {
                    if let ExprType::Name(t) = target {
                        // `a = a` is not an alias.
                        if t.id != source.id {
                            self.containers.insert(t.id.clone());
                            self.aliases
                                .push((t.id.clone(), source.id.clone(), lineno));
                        }
                    }
                }
            }
        }
        // Subscript stores (`a[i] = v`) mutate the container; a bare-name
        // target is a REBINDING, which never touches the shared object.
        for target in &a.targets {
            if let ExprType::Subscript(_) = target {
                if let Some(name) = root_name_of(target) {
                    self.mutated.insert(name.to_string());
                }
            }
        }
        self.visit_expr(&a.value);
    }

    fn visit_call(&mut self, c: &crate::Call, lineno: Option<usize>) {
        // Mutating method call on a container name: `a.append(x)`,
        // `a.sort()`, ...
        if let ExprType::Attribute(attr) = c.func.as_ref() {
            if let Some(receiver) = root_name_of(&attr.value) {
                if MUTATING_METHODS.contains(&attr.attr.as_str())
                    && self.containers.contains(receiver)
                {
                    self.mutated.insert(receiver.to_string());
                }
            }
        }
        // Shape 2: container names passed to a function that mutates the
        // corresponding parameter, when the name is read again afterwards.
        if let ExprType::Name(callee) = c.func.as_ref() {
            if let Some(SymbolTableNode::FunctionDef(func)) = self.symbols.get(&callee.id) {
                for (i, arg) in c.args.iter().enumerate() {
                    if let ExprType::Name(n) = arg {
                        if self.containers.contains(&n.id)
                            && function_mutates_param(func, i)
                            && self.use_counts.get(&n.id).copied().unwrap_or(0) > 1
                        {
                            self.alias_calls.push((n.id.clone(), lineno));
                        }
                    }
                }
            }
        }
        // Recurse into the call's subexpressions for nested mutations.
        self.visit_expr(&c.func);
        for arg in &c.args {
            self.visit_expr(arg);
        }
        for kw in &c.keywords {
            self.visit_expr(&kw.value);
        }
    }

    /// Recursively scan an expression for nested mutating calls and
    /// shape-2 calls (e.g. `f(a.append(1))`, `g(xs)` inside a larger
    /// expression).
    fn visit_expr(&mut self, e: &ExprType) {
        match e {
            ExprType::Call(c) => self.visit_call(c, None),
            ExprType::BoolOp(b) => {
                for v in &b.values {
                    self.visit_expr(v);
                }
            }
            ExprType::BinOp(b) => {
                self.visit_expr(&b.left);
                self.visit_expr(&b.right);
            }
            ExprType::UnaryOp(u) => self.visit_expr(&u.operand),
            ExprType::IfExp(i) => {
                self.visit_expr(&i.test);
                self.visit_expr(&i.body);
                self.visit_expr(&i.orelse);
            }
            ExprType::Dict(d) => {
                for k in &d.keys {
                    if let Some(k) = k {
                        self.visit_expr(k);
                    }
                }
                for v in &d.values {
                    self.visit_expr(v);
                }
            }
            ExprType::Set(s) => {
                for elt in &s.elts {
                    self.visit_expr(elt);
                }
            }
            ExprType::List(items) => {
                for item in items {
                    self.visit_expr(item);
                }
            }
            ExprType::Tuple(t) => {
                for elt in &t.elts {
                    self.visit_expr(elt);
                }
            }
            ExprType::Compare(c) => {
                self.visit_expr(&c.left);
                for r in &c.comparators {
                    self.visit_expr(r);
                }
            }
            ExprType::Attribute(a) => self.visit_expr(&a.value),
            ExprType::Subscript(s) => {
                self.visit_expr(&s.value);
                match &s.kind {
                    crate::SubscriptKind::Index(e) => self.visit_expr(e),
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        if let Some(l) = lower {
                            self.visit_expr(l);
                        }
                        if let Some(u) = upper {
                            self.visit_expr(u);
                        }
                        if let Some(s) = step {
                            self.visit_expr(s);
                        }
                    }
                }
            }
            ExprType::Starred(s) => self.visit_expr(&s.value),
            ExprType::NamedExpr(n) => {
                self.visit_expr(&n.left);
                self.visit_expr(&n.right);
            }
            ExprType::Await(a) => self.visit_expr(&a.value),
            ExprType::Yield(y) => {
                if let Some(v) = &y.value {
                    self.visit_expr(v);
                }
            }
            ExprType::YieldFrom(y) => self.visit_expr(&y.value),
            ExprType::Lambda(l) => self.visit_expr(&l.body),
            ExprType::JoinedStr(f) => {
                for v in &f.values {
                    self.visit_expr(v);
                }
            }
            ExprType::FormattedValue(f) => {
                self.visit_expr(&f.value);
                if let Some(spec) = &f.format_spec {
                    self.visit_expr(spec);
                }
            }
            ExprType::ListComp(l) => self.visit_comprehension(&l.elt, &l.generators),
            ExprType::SetComp(s) => self.visit_comprehension(&s.elt, &s.generators),
            ExprType::DictComp(d) => self.visit_comprehension(&d.value, &d.generators),
            ExprType::GeneratorExp(g) => self.visit_comprehension(&g.elt, &g.generators),
            ExprType::Name(_)
            | ExprType::Constant(_)
            | ExprType::NoneType(_)
            | ExprType::Unknown
            | ExprType::Unimplemented(_) => {}
        }
    }

    fn visit_comprehension(
        &mut self,
        elt: &ExprType,
        generators: &[crate::Comprehension],
    ) {
        self.visit_expr(elt);
        for generator in generators {
            self.visit_expr(&generator.iter);
            for cond in &generator.ifs {
                self.visit_expr(cond);
            }
        }
    }
}

/// Whether a user function mutates its `index`-th positional parameter
/// anywhere in its body (directly — no transitivity through calls it makes,
/// which keeps this conservative: it only fires on shapes the current model
/// provably gets wrong).
fn function_mutates_param(func: &FunctionDef, index: usize) -> bool {
    let Some(param) = func
        .args
        .posonlyargs
        .iter()
        .chain(func.args.args.iter())
        .nth(index)
    else {
        return false;
    };
    let name = &param.arg;
    let mut found = false;
    scan_mutations(&func.body, name, &mut found);
    found
}

fn scan_mutations(body: &[Statement], name: &str, found: &mut bool) {
    if *found {
        return;
    }
    for stmt in body {
        if *found {
            return;
        }
        match &stmt.statement {
            StatementType::Assign(a) => {
                // `param[i] = v` mutates the container; a bare-name target
                // rebinds the local parameter, which does not touch the
                // caller's object.
                for target in &a.targets {
                    if let ExprType::Subscript(_) = target {
                        if root_name_of(target) == Some(name) {
                            *found = true;
                        }
                    }
                }
            }
            StatementType::AugAssign(a) => {
                if root_name_of(&a.target) == Some(name) {
                    *found = true;
                }
            }
            StatementType::Call(c) => {
                if let ExprType::Attribute(attr) = c.func.as_ref() {
                    if root_name_of(&attr.value) == Some(name)
                        && MUTATING_METHODS.contains(&attr.attr.as_str())
                    {
                        *found = true;
                    }
                }
            }
            StatementType::Expr(e) => {
                if let ExprType::Call(c) = &e.value {
                    if let ExprType::Attribute(attr) = c.func.as_ref() {
                        if root_name_of(&attr.value) == Some(name)
                            && MUTATING_METHODS.contains(&attr.attr.as_str())
                        {
                            *found = true;
                        }
                    }
                }
            }
            StatementType::If(i) => {
                scan_mutations(&i.body, name, found);
                scan_mutations(&i.orelse, name, found);
            }
            StatementType::For(f) => {
                scan_mutations(&f.body, name, found);
                scan_mutations(&f.orelse, name, found);
            }
            StatementType::AsyncFor(f) => {
                scan_mutations(&f.body, name, found);
                scan_mutations(&f.orelse, name, found);
            }
            StatementType::While(w) => {
                scan_mutations(&w.body, name, found);
                scan_mutations(&w.orelse, name, found);
            }
            StatementType::Try(t) => {
                scan_mutations(&t.body, name, found);
                for handler in &t.handlers {
                    scan_mutations(&handler.body, name, found);
                }
                scan_mutations(&t.orelse, name, found);
                scan_mutations(&t.finalbody, name, found);
            }
            StatementType::With(w) => scan_mutations(&w.body, name, found),
            StatementType::AsyncWith(w) => scan_mutations(&w.body, name, found),
            _ => {}
        }
    }
}
