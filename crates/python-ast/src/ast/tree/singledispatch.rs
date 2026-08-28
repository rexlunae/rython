//! Conversion-time desugaring of `functools.singledispatch` (issue #181).
//!
//! Python's single-dispatch generic function is a family of definitions
//! bound to one name:
//!
//! ```python
//! @functools.singledispatch
//! def yield_lines(iterable):
//!     return itertools.chain.from_iterable(map(yield_lines, iterable))
//!
//! @yield_lines.register(str)
//! def _(text):
//!     return filter(_nonblank, map(str.strip, text.splitlines()))
//! ```
//!
//! At runtime CPython picks an implementation from the first argument's
//! type. rython's value model has no runtime types to dispatch on — but
//! it already monomorphizes `isinstance`-dispatching functions
//! ([`crate::ast::tree::specialize`]), emitting one Rust function per
//! input type and routing each call site by its argument's static type.
//! That is exactly the machinery singledispatch needs, so this pass does
//! not lower anything itself: it REWRITES the family into the single
//! `isinstance` chain that expresses the same dispatch,
//!
//! ```python
//! def yield_lines(iterable):
//!     if isinstance(iterable, str):
//!         text = iterable
//!         return filter(_nonblank, map(str.strip, text.splitlines()))
//!     return itertools.chain.from_iterable(map(yield_lines, iterable))
//! ```
//!
//! and lets the existing pass specialize it. Inside the `str` variant the
//! parameter is a real `String`, so the specialization body gets ordinary
//! `str` methods rather than method calls on a boxed value — the reason
//! this shape is expressible at all.
//!
//! Divergences (recorded in docs/spec.md §12):
//! - dispatch is FIRST-MATCH IN REGISTRATION ORDER, not CPython's MRO
//!   walk. For the disjoint concrete types real code registers the two
//!   agree; a registration on a base class followed by one on its
//!   subclass would pick the base where CPython picks the subclass.
//! - `<generic>.register` is resolved SYNTACTICALLY, so the generic and
//!   all of its specializations must live in one module.
//!
//! Loud boundaries (never silently different):
//! - a `@x.register(...)` shape this pass cannot read (a bare
//!   `@x.register` taking the type from an annotation, a non-Name
//!   dispatch type, a `register(T, impl)` call);
//! - a register whose generic is not a `@singledispatch` definition in
//!   the same module;
//! - a specialization whose parameter list does not match the generic's.

use crate::ast::tree::decorator::{parse_decorator, Decorator};
use crate::{
    Assign, Call, ExprType, FunctionDef, If, Name, Statement, StatementType,
};

/// One registered specialization, in registration order.
struct Registration {
    dispatch_type: String,
    def: FunctionDef,
    stmt_index: usize,
}

/// Whether a definition carries `@functools.singledispatch`.
fn is_generic(def: &FunctionDef) -> bool {
    matches!(
        parse_decorator(&def.decorator_list),
        Ok(Some(Decorator::SingleDispatch))
    )
}

/// The `@<generic>.register(T)` a definition carries, if any.
fn registration_of(def: &FunctionDef) -> Option<(String, String)> {
    match parse_decorator(&def.decorator_list) {
        Ok(Some(Decorator::Register {
            generic,
            dispatch_type,
        })) => Some((generic, dispatch_type)),
        _ => None,
    }
}

/// A `.register` decorator this pass could not read — the loud boundary.
/// Matched on the raw expression so the shapes `parse_decorator` turns
/// away (a bare `@x.register`, `@x.register(T, impl)`, a dotted dispatch
/// type) are still recognized as singledispatch registrations rather
/// than falling through to the generic unsupported-decorator error.
fn unreadable_register(d: &ExprType) -> Option<String> {
    let attr = match d {
        ExprType::Attribute(a) => a,
        ExprType::Call(c) => match c.func.as_ref() {
            ExprType::Attribute(a) => a,
            _ => return None,
        },
        _ => return None,
    };
    if attr.attr != "register" {
        return None;
    }
    match attr.value.as_ref() {
        ExprType::Name(n) => Some(n.id.clone()),
        _ => None,
    }
}

/// Rewrite every `@functools.singledispatch` family in a module body into
/// the equivalent `isinstance` chain. Bodies with no singledispatch are
/// returned untouched.
pub fn desugar_module(body: Vec<Statement>) -> Result<Vec<Statement>, String> {
    let generics: Vec<String> = body
        .iter()
        .filter_map(|s| match &s.statement {
            StatementType::FunctionDef(f) if is_generic(f) => Some(f.name.clone()),
            _ => None,
        })
        .collect();
    if generics.is_empty() {
        // A `.register` with no generic in the module is still a loud
        // boundary: the family cannot be assembled from one module.
        for s in &body {
            if let StatementType::FunctionDef(f) = &s.statement
                && let Some(name) = f.decorator_list.iter().find_map(unreadable_register)
            {
                return Err(format!(
                    "`@{name}.register` decorates `{}`, but `{name}` is not a \
                     `@functools.singledispatch` definition in this module; rython \
                     assembles a singledispatch family syntactically, so the generic \
                     and every specialization must live in one module",
                    f.name
                ));
            }
        }
        return Ok(body);
    }

    // Collect the registrations, keyed by the generic they specialize.
    let mut registrations: Vec<(String, Registration)> = Vec::new();
    for (stmt_index, s) in body.iter().enumerate() {
        let StatementType::FunctionDef(f) = &s.statement else {
            continue;
        };
        if let Some((generic, dispatch_type)) = registration_of(f) {
            if !generics.contains(&generic) {
                return Err(format!(
                    "`@{generic}.register({dispatch_type})` decorates `{}`, but \
                     `{generic}` is not a `@functools.singledispatch` definition in \
                     this module; rython assembles a singledispatch family \
                     syntactically, so the generic and every specialization must live \
                     in one module",
                    f.name
                ));
            }
            registrations.push((
                generic,
                Registration {
                    dispatch_type,
                    def: f.clone(),
                    stmt_index,
                },
            ));
            continue;
        }
        // A `.register` shape parse_decorator could not read, on a
        // generic this module defines: refuse rather than drop it.
        if let Some(name) = f.decorator_list.iter().find_map(unreadable_register)
            && generics.contains(&name)
        {
            return Err(format!(
                "`@{name}.register` on `{}` is not in a form rython can read; only \
                 `@{name}.register(<type>)` with a single bare type name is \
                 supported (the annotation-typed and two-argument forms are not)",
                f.name
            ));
        }
    }

    let consumed: Vec<usize> = registrations.iter().map(|(_, r)| r.stmt_index).collect();
    let mut out = Vec::with_capacity(body.len());
    for (index, s) in body.into_iter().enumerate() {
        if consumed.contains(&index) {
            continue;
        }
        let StatementType::FunctionDef(f) = &s.statement else {
            out.push(s);
            continue;
        };
        if !is_generic(f) {
            out.push(s);
            continue;
        }
        let specializations: Vec<&Registration> = registrations
            .iter()
            .filter(|(g, _)| *g == f.name)
            .map(|(_, r)| r)
            .collect();
        let fused = fuse(f, &specializations, &s)?;
        out.push(Statement {
            statement: StatementType::FunctionDef(fused),
            ..s
        });
    }
    Ok(out)
}

/// The generic's positional parameter names, or a loud error for the
/// shapes dispatch cannot express.
fn positional_names(def: &FunctionDef, what: &str) -> Result<Vec<String>, String> {
    if def.args.vararg.is_some() || def.args.kwarg.is_some() {
        return Err(format!(
            "singledispatch {what} `{}` takes *args/**kwargs; rython dispatches on \
             a fixed positional parameter list",
            def.name
        ));
    }
    Ok(def
        .args
        .posonlyargs
        .iter()
        .chain(def.args.args.iter())
        .map(|p| p.arg.clone())
        .collect())
}

/// Build the generic's fused body: one `isinstance` arm per registered
/// specialization, in registration order, then the decorated definition's
/// own body as the fallthrough default.
fn fuse(
    generic: &FunctionDef,
    specializations: &[&Registration],
    at: &Statement,
) -> Result<FunctionDef, String> {
    let params = positional_names(generic, "generic")?;
    // CPython's LAST registration of a type wins; a fused isinstance
    // chain takes the FIRST. Rather than pick one silently, refuse the
    // shape — no real family registers the same type twice.
    for (i, reg) in specializations.iter().enumerate() {
        if let Some(prev) = specializations[..i]
            .iter()
            .find(|p| p.dispatch_type == reg.dispatch_type)
        {
            return Err(format!(
                "`{}` is registered for `{}` twice (`{}` and `{}`); CPython's later \
                 registration wins, and rython's fused dispatch would take the earlier \
                 one — remove one of them",
                generic.name, reg.dispatch_type, prev.def.name, reg.def.name
            ));
        }
    }
    if params.is_empty() {
        return Err(format!(
            "`@functools.singledispatch` on `{}` takes no arguments; the dispatch \
             reads the first one",
            generic.name
        ));
    }
    let axis = params[0].clone();

    let mut body: Vec<Statement> = Vec::new();
    for reg in specializations {
        let spec_params = positional_names(&reg.def, "specialization")?;
        if spec_params.len() != params.len() {
            return Err(format!(
                "`@{}.register({})` defines `{}` with {} parameter(s) but the generic \
                 takes {}; rython fuses the family into one function, so the \
                 specializations must share the generic's signature",
                generic.name,
                reg.dispatch_type,
                reg.def.name,
                spec_params.len(),
                params.len()
            ));
        }
        // Bind the specialization's own parameter names to the generic's,
        // so its body reads unchanged. Identical names need no binding.
        let mut arm: Vec<Statement> = Vec::new();
        for (spec, outer) in spec_params.iter().zip(params.iter()) {
            if spec == outer {
                continue;
            }
            arm.push(stmt(
                at,
                StatementType::Assign(Assign {
                    targets: vec![name_expr(spec)],
                    value: name_expr(outer),
                    type_comment: None,
                    annotation: None,
                }),
            ));
        }
        arm.extend(reg.def.body.iter().cloned());
        body.push(stmt(
            at,
            StatementType::If(If {
                test: isinstance_call(&axis, &reg.dispatch_type),
                body: arm,
                orelse: Vec::new(),
                lineno: at.lineno,
                col_offset: at.col_offset,
                end_lineno: at.end_lineno,
                end_col_offset: at.end_col_offset,
            }),
        ));
    }
    body.extend(generic.body.iter().cloned());

    Ok(FunctionDef {
        name: generic.name.clone(),
        args: generic.args.clone(),
        body,
        // The family is fully expressed by the fused body; the
        // singledispatch marker itself has no residual meaning.
        decorator_list: Vec::new(),
        returns: generic.returns.clone(),
    })
}

/// `isinstance(<axis>, <ty>)`.
fn isinstance_call(axis: &str, ty: &str) -> ExprType {
    ExprType::Call(Call {
        func: Box::new(name_expr("isinstance")),
        args: vec![name_expr(axis), name_expr(ty)],
        keywords: Vec::new(),
    })
}

fn name_expr(id: &str) -> ExprType {
    ExprType::Name(Name { id: id.to_string() })
}

/// A synthesized statement carrying the family's source position, so
/// errors raised inside a fused body still point at the user's Python.
fn stmt(at: &Statement, statement: StatementType) -> Statement {
    Statement {
        lineno: at.lineno,
        col_offset: at.col_offset,
        end_lineno: at.end_lineno,
        end_col_offset: at.end_col_offset,
        statement,
    }
}
