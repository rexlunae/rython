//! Reads of a runtime module's name that Python never bound (issue #435).
//!
//! Every generated module glob-imports the stdpython runtime
//! (`use stdpython::*`), which exports the runtime modules (`math`, `os`,
//! `json`, ...) by their Python names. A read of `math` that the program
//! never bound therefore resolves to the runtime module in Rust, where
//! CPython raises `NameError`: `import math as m` binds only `m`, so a
//! later `math.sqrt(9.0)` runs (and prints) instead of failing. Such a
//! read is a conversion error.
//!
//! A name counts as bound when its scope or an enclosing one binds it
//! anywhere (an import, an assignment, a def, a parameter, a `global`
//! declaration...), following Python's scoping: a function sees its own
//! bindings, its enclosing functions' and the module's; a class body sees
//! its own and the outer ones, but the methods in it do not see the class
//! body's. Where the scope binds the name is not checked (a read before
//! the binding is the deleted-name and flow checks' business), and a
//! `from m import *` turns the check off for the module (it may bind any
//! name). Annotations are not checked: they may name a module bound only
//! under `TYPE_CHECKING` or never evaluated (`from __future__ import
//! annotations`).

use std::collections::HashSet;

use crate::ast::tree::std_module::StdModule;
use crate::ast::tree::visit::{self, Bindings, Descend, Flow};
use crate::{ExprType, Statement, StatementType};

/// Every name `body` binds in its own scope (nested definitions bind
/// their own names here, not their bodies'), over-approximated: a
/// comprehension target or a lambda parameter counts too, which can only
/// hide an unbound read, never invent one.
fn scope_bindings(body: &[Statement]) -> HashSet<String> {
    let mut names = HashSet::new();
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        names.extend(visit::stmt_bound_names(s, Bindings::Every));
        Flow::Continue
    });
    names
}

/// The parameter names of a `def`.
fn params(f: &crate::FunctionDef) -> impl Iterator<Item = String> + '_ {
    let a = &f.args;
    a.posonlyargs
        .iter()
        .chain(a.args.iter())
        .chain(a.kwonlyargs.iter())
        .chain(a.vararg.iter())
        .chain(a.kwarg.iter())
        .map(|p| p.arg.clone())
}

/// The expressions a statement evaluates in its own scope, annotations
/// left out (see the module docs).
fn evaluated_exprs(s: &Statement) -> Vec<&ExprType> {
    match &s.statement {
        StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => f
            .decorator_list
            .iter()
            .chain(f.args.defaults.iter().map(|d| d.as_ref()))
            .chain(f.args.kw_defaults.iter().flatten().map(|d| d.as_ref()))
            .collect(),
        _ => visit::stmt_exprs(s),
    }
}

/// The first read of an unbound runtime module name in `e`.
fn unbound_read<'a>(e: &'a ExprType, visible: &HashSet<String>) -> Option<&'a str> {
    let mut found = None;
    visit::any_expr(e, |sub| {
        if let ExprType::Name(n) = sub
            && StdModule::from_name(&n.id).is_some()
            && !visible.contains(&n.id)
        {
            found = Some(n.id.as_str());
            return true;
        }
        false
    });
    found
}

/// Check one scope's statements: their expressions against `visible`,
/// each nested `def` against `enclosing` (what a function nested here
/// sees: a class body's own names are not among them) plus its own
/// bindings, each nested class body against `enclosing` plus its own.
fn check_scope(
    body: &[Statement],
    visible: &HashSet<String>,
    enclosing: &HashSet<String>,
) -> Result<(), String> {
    let mut err = None;
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        let result = (|| {
            for e in evaluated_exprs(s) {
                if let Some(name) = unbound_read(e, visible) {
                    return Err(refusal(name, s.lineno));
                }
            }
            match &s.statement {
                StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                    let mut inner = enclosing.clone();
                    inner.extend(params(f));
                    inner.extend(scope_bindings(&f.body));
                    check_scope(&f.body, &inner, &inner)
                }
                StatementType::ClassDef(c) => {
                    let mut class_scope = enclosing.clone();
                    class_scope.extend(scope_bindings(&c.body));
                    check_scope(&c.body, &class_scope, enclosing)
                }
                _ => Ok(()),
            }
        })();
        match result {
            Ok(()) => Flow::Continue,
            Err(e) => {
                err = Some(e);
                Flow::Stop
            }
        }
    });
    err.map_or(Ok(()), Err)
}

fn refusal(name: &str, line: Option<usize>) -> String {
    let at = line.map(|l| format!(" (line {})", l)).unwrap_or_default();
    format!(
        "`{name}`{at} is read but never bound in this scope: CPython raises \
         NameError here, but the generated code would resolve it to rython's \
         runtime `{name}` module and run on; add `import {name}` (an aliased \
         `import {name} as m` binds only `m`) — rython refuses to silently run \
         code CPython rejects (issue #435)"
    )
}

/// Refuse every read of a runtime module name that `body` (a whole
/// module) never binds where the read can see it.
pub(crate) fn check_module(body: &[Statement]) -> Result<(), String> {
    let star_import = visit::any_stmt(
        body,
        Descend::All,
        |s| matches!(&s.statement, StatementType::ImportFrom(im) if im.names.iter().any(|a| a.name == "*")),
    );
    if star_import {
        return Ok(());
    }
    let mut module = scope_bindings(body);
    // A `global` declaration in a function binds the name at module scope
    // when the function stores it (`global math; import math`).
    visit::walk_stmts(body, Descend::All, &mut |s| {
        if let StatementType::Global(names) = &s.statement {
            module.extend(names.iter().cloned());
        }
        Flow::Continue
    });
    check_scope(body, &module, &module)
}
