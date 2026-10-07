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
//! (an import, an assignment, a def, a parameter, an `except ... as`, a
//! `global` declaration in a function that stores the name...), following
//! Python's scoping: a function sees its own bindings, its enclosing
//! functions' and the module's; a class body sees its own and the outer
//! ones, but the functions, lambdas and comprehensions in it do not see
//! the class body's. A comprehension's targets and a lambda's parameters
//! bind only inside it. Where the scope binds the name is not checked (a
//! read before the binding is the deleted-name and flow checks'
//! business). A `from m import *` of a module other than a runtime module
//! turns the check off for the module (it may bind any name; a runtime
//! module's star exports are its members, never a module name).
//! Annotations are not checked: they may name a module bound only under
//! `TYPE_CHECKING` or never evaluated (`from __future__ import
//! annotations`).

use std::collections::HashSet;

use crate::ast::tree::std_module::StdModule;
use crate::ast::tree::visit::{self, Bindings, Descend, Flow};
use crate::{ExprType, Statement, StatementType};

type Names = HashSet<String>;

/// The names `body` binds in its own scope: what each statement binds
/// where it sits (nested definitions bind their own names here, not their
/// bodies' or their parameters'; a comprehension's walrus binds here, its
/// targets do not) plus `except ... as` names.
fn scope_bindings(body: &[Statement]) -> Names {
    let mut names = Names::new();
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        names.extend(visit::stmt_bound_names(s, Bindings::Scope));
        if let StatementType::Try(t) = &s.statement {
            names.extend(t.handlers.iter().filter_map(|h| h.name.clone()));
        }
        Flow::Continue
    });
    names
}

/// The parameter names of a `def` or a lambda.
fn params(a: &crate::ParameterList) -> impl Iterator<Item = String> + '_ {
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
    let mut out: Vec<&ExprType> = match &s.statement {
        StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => f
            .decorator_list
            .iter()
            .chain(f.args.defaults.iter().map(|d| d.as_ref()))
            .chain(f.args.kw_defaults.iter().flatten().map(|d| d.as_ref()))
            .collect(),
        _ => visit::stmt_exprs(s),
    };
    if let StatementType::Try(t) = &s.statement {
        out.extend(t.handlers.iter().filter_map(|h| h.exception_type.as_ref()));
    }
    out
}

/// The first read of an unbound runtime module name in `e`, which sits
/// in a scope seeing `visible`. A lambda body or a comprehension (past
/// its first iterable, which the enclosing scope evaluates) is a scope of
/// its own: it sees `enclosing` (a class body's names are not in it) plus
/// its own parameters or targets.
fn unbound_read<'a>(e: &'a ExprType, visible: &Names, enclosing: &Names) -> Option<&'a str> {
    match e {
        ExprType::Name(n) if StdModule::from_name(&n.id).is_some() && !visible.contains(&n.id) => {
            return Some(n.id.as_str());
        }
        ExprType::Lambda(l) => {
            let defaults = l.args.defaults.iter().map(|d| d.as_ref());
            let kw_defaults = l.args.kw_defaults.iter().flatten().map(|d| d.as_ref());
            if let Some(name) = defaults
                .chain(kw_defaults)
                .find_map(|d| unbound_read(d, visible, enclosing))
            {
                return Some(name);
            }
            let mut inner = enclosing.clone();
            inner.extend(params(&l.args));
            return unbound_read(&l.body, &inner, &inner);
        }
        _ => {}
    }
    if let Some((elts, generators)) = comprehension(e) {
        let first = generators.first()?;
        if let Some(name) = unbound_read(&first.iter, visible, enclosing) {
            return Some(name);
        }
        let mut inner = enclosing.clone();
        for g in generators {
            inner.extend(
                visit::target_names(&g.target)
                    .into_iter()
                    .map(str::to_string),
            );
        }
        let parts = generators
            .iter()
            .enumerate()
            .flat_map(|(i, g)| (i > 0).then_some(&g.iter).into_iter().chain(g.ifs.iter()))
            .chain(elts);
        return parts
            .into_iter()
            .find_map(|x| unbound_read(x, &inner, &inner));
    }
    let mut found = None;
    visit::each_subexpr(e, Descend::All, &mut |sub| {
        found = unbound_read(sub, visible, enclosing);
        found.is_none()
    });
    found
}

/// A comprehension's element expressions and generators, when `e` is one.
fn comprehension(e: &ExprType) -> Option<(Vec<&ExprType>, &[crate::Comprehension])> {
    match e {
        ExprType::ListComp(c) => Some((vec![c.elt.as_ref()], &c.generators)),
        ExprType::GeneratorExp(c) => Some((vec![c.elt.as_ref()], &c.generators)),
        ExprType::SetComp(c) => Some((vec![c.elt.as_ref()], &c.generators)),
        ExprType::DictComp(c) => Some((vec![c.key.as_ref(), c.value.as_ref()], &c.generators)),
        _ => None,
    }
}

/// Check one scope's statements: their expressions against `visible`,
/// with `enclosing` what a function, lambda or comprehension nested here
/// sees (a class body's own names are not among them); each nested `def`
/// against `enclosing` plus its own parameters and bindings, each nested
/// class body against `enclosing` plus its own bindings.
fn check_scope(body: &[Statement], visible: &Names, enclosing: &Names) -> Result<(), String> {
    let mut err = None;
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        let result = (|| {
            for e in evaluated_exprs(s) {
                if let Some(name) = unbound_read(e, visible, enclosing) {
                    return Err(refusal(name, s.lineno));
                }
            }
            match &s.statement {
                StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                    let mut inner = enclosing.clone();
                    inner.extend(params(&f.args));
                    inner.extend(scope_bindings(&f.body));
                    check_scope(&f.body, &inner, &inner)
                }
                StatementType::ClassDef(c) => {
                    let mut class_scope = visible.clone();
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
    // A star import of anything but a runtime module may bind any name.
    let opaque_star_import = visit::any_stmt(body, Descend::All, |s| {
        matches!(&s.statement, StatementType::ImportFrom(im)
            if im.names.iter().any(|a| a.name == "*")
                && !(im.level == 0
                    && StdModule::from_name(im.module.split('.').next().unwrap_or("")).is_some()))
    });
    if opaque_star_import {
        return Ok(());
    }
    let mut module = scope_bindings(body);
    // A function's `global` declaration binds the name at module scope
    // when that function stores it (`global math; import math`); a bare
    // declaration binds nothing.
    visit::walk_stmts(body, Descend::All, &mut |s| {
        if let StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) = &s.statement {
            let stored = scope_bindings(&f.body);
            visit::walk_stmts(&f.body, Descend::SkipDefs, &mut |inner| {
                if let StatementType::Global(names) = &inner.statement {
                    module.extend(names.iter().filter(|n| stored.contains(*n)).cloned());
                }
                Flow::Continue
            });
        }
        Flow::Continue
    });
    check_scope(body, &module, &module)
}
