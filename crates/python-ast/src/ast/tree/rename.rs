//! Scoped receiver renaming for methods whose first parameter is not
//! named `self` (Python binds the instance to the FIRST parameter
//! whatever its name — boto3's `factory_self`; issue #132).
//!
//! The Rust receiver is always `self`, and large parts of codegen
//! special-case the literal name, so before lowering a method whose
//! receiver has another name, every reference to that name in the method
//! body is rewritten to `self`. The rewrite is scope-aware:
//!
//! - a nested `def`/`lambda`/comprehension that BINDS the target name as
//!   its own parameter or loop variable keeps its scope untouched;
//! - a nested function that does NOT bind it captures the enclosing
//!   receiver, so its body is renamed too;
//! - rebinding the receiver (`factory_self = ...`) is not expressible on
//!   a `&self` receiver — loud conversion error rather than silently
//!   different codegen.

use super::expression::ExprType;
use super::statement::{Statement, StatementType};
use super::*;

/// Rewrite `from` -> `to` across a statement list. Fails loudly when the
/// body rebinds the receiver name directly.
pub(crate) fn rename_receiver_in_body(
    body: &[Statement],
    from: &str,
    to: &str,
) -> Result<Vec<Statement>, String> {
    body.iter().map(|s| rename_statement(s, from, to)).collect()
}

fn rename_statement(stmt: &Statement, from: &str, to: &str) -> Result<Statement, String> {
    let statement = match &stmt.statement {
        StatementType::Assign(a) => {
            // Rebinding the receiver name itself cannot live on `&self`.
            for target in &a.targets {
                if matches!(target, ExprType::Name(n) if n.id == from) {
                    return Err(format!(
                        "method rebinds its receiver `{from}`; rython's receiver is \
                         an immutable `&self`, so the reassignment has no lowering"
                    ));
                }
            }
            StatementType::Assign(Assign {
                targets: a.targets.iter().map(|t| rename_expr(t, from, to)).collect(),
                value: rename_expr(&a.value, from, to),
                type_comment: a.type_comment.clone(),
                annotation: a.annotation.clone(),
            })
        }
        StatementType::AugAssign(a) => {
            if matches!(&a.target, ExprType::Name(n) if n.id == from) {
                return Err(format!(
                    "method aug-rebinds its receiver `{from}`; rython's receiver is \
                     an immutable `&self`, so the reassignment has no lowering"
                ));
            }
            let mut a = a.clone();
            a.target = rename_expr(&a.target, from, to);
            StatementType::AugAssign(a)
        }
        StatementType::Return(r) => StatementType::Return(match r {
            Some(e) => Some(Expr {
                value: rename_expr(&e.value, from, to),
                ctx: e.ctx.clone(),
                lineno: e.lineno,
                col_offset: e.col_offset,
                end_lineno: e.end_lineno,
                end_col_offset: e.end_col_offset,
            }),
            None => None,
        }),
        StatementType::Expr(e) => StatementType::Expr(Expr {
            value: rename_expr(&e.value, from, to),
            ctx: e.ctx.clone(),
            lineno: e.lineno,
            col_offset: e.col_offset,
            end_lineno: e.end_lineno,
            end_col_offset: e.end_col_offset,
        }),
        StatementType::If(i) => StatementType::If(If {
            test: rename_expr(&i.test, from, to),
            body: rename_receiver_in_body(&i.body, from, to)?,
            orelse: rename_receiver_in_body(&i.orelse, from, to)?,
            lineno: i.lineno,
            col_offset: i.col_offset,
            end_lineno: i.end_lineno,
            end_col_offset: i.end_col_offset,
        }),
        StatementType::While(w) => StatementType::While(super::while_stmt::While {
            test: rename_expr(&w.test, from, to),
            body: rename_receiver_in_body(&w.body, from, to)?,
            orelse: rename_receiver_in_body(&w.orelse, from, to)?,
            lineno: w.lineno,
            col_offset: w.col_offset,
            end_lineno: w.end_lineno,
            end_col_offset: w.end_col_offset,
        }),
        StatementType::For(f) => {
            // A for-target binding the receiver name shadows it inside the
            // loop — leave that scope alone.
            let binds = expr_binds_name(&f.target, from);
            let body = if binds {
                f.body.clone()
            } else {
                rename_receiver_in_body(&f.body, from, to)?
            };
            let orelse = if binds {
                f.orelse.clone()
            } else {
                rename_receiver_in_body(&f.orelse, from, to)?
            };
            StatementType::For(super::for_stmt::For {
                target: f.target.clone(),
                iter: rename_expr(&f.iter, from, to),
                body,
                orelse,
                lineno: f.lineno,
                col_offset: f.col_offset,
                end_lineno: f.end_lineno,
                end_col_offset: f.end_col_offset,
            })
        }
        StatementType::Try(t) => StatementType::Try(Try {
            body: rename_receiver_in_body(&t.body, from, to)?,
            handlers: {
                let mut handlers = Vec::with_capacity(t.handlers.len());
                for h in &t.handlers {
                    let mut h = h.clone();
                    // An `except ... as <name>:` handler binding the
                    // receiver name shadows it inside the handler body.
                    if h.name.as_deref() != Some(from) {
                        h.body = rename_receiver_in_body(&h.body, from, to)?;
                    }
                    handlers.push(h);
                }
                handlers
            },
            orelse: rename_receiver_in_body(&t.orelse, from, to)?,
            finalbody: rename_receiver_in_body(&t.finalbody, from, to)?,
            lineno: t.lineno,
            col_offset: t.col_offset,
            end_lineno: t.end_lineno,
            end_col_offset: t.end_col_offset,
        }),
        StatementType::With(w) => {
            let binds = w.items.iter().any(|item| {
                item.optional_vars
                    .as_ref()
                    .is_some_and(|v| expr_binds_name(v, from))
            });
            StatementType::With(With {
                items: w
                    .items
                    .iter()
                    .map(|item| WithItem {
                        context_expr: rename_expr(&item.context_expr, from, to),
                        optional_vars: item
                            .optional_vars
                            .as_ref()
                            .map(|v| rename_expr(v, from, to)),
                    })
                    .collect(),
                body: if binds {
                    w.body.clone()
                } else {
                    rename_receiver_in_body(&w.body, from, to)?
                },
                lineno: w.lineno,
                col_offset: w.col_offset,
                end_lineno: w.end_lineno,
                end_col_offset: w.end_col_offset,
            })
        }
        StatementType::Raise(r) => StatementType::Raise(super::raise_stmt::Raise {
            exc: r.exc.as_ref().map(|e| rename_expr(e, from, to)),
            cause: r.cause.as_ref().map(|c| rename_expr(c, from, to)),
            lineno: r.lineno,
            col_offset: r.col_offset,
            end_lineno: r.end_lineno,
            end_col_offset: r.end_col_offset,
        }),
        StatementType::Assert { test, msg } => StatementType::Assert {
            test: Box::new(rename_expr(test.as_ref(), from, to)),
            msg: match msg {
                Some(m) => Some(Box::new(rename_expr(m.as_ref(), from, to))),
                None => None,
            },
        },
        // `del <receiver>` unbinds the binding — same no-lowering posture
        // as rebinding it; index targets rename through.
        StatementType::Delete(targets) => {
            for target in targets {
                if matches!(target, ExprType::Name(n) if n.id == from) {
                    return Err(format!(
                        "method deletes its receiver `{from}`; rython's receiver is \
                         an immutable `&self`, so the unbinding has no lowering"
                    ));
                }
            }
            let mut renamed = Vec::with_capacity(targets.len());
            for t in targets {
                renamed.push(rename_expr(t, from, to));
            }
            StatementType::Delete(renamed)
        }
        StatementType::AsyncFor(f) => {
            let binds = expr_binds_name(&f.target, from);
            if binds {
                stmt.statement.clone()
            } else {
                let mut f = f.clone();
                f.iter = rename_expr(&f.iter, from, to);
                f.body = rename_receiver_in_body(&f.body, from, to)?;
                f.orelse = rename_receiver_in_body(&f.orelse, from, to)?;
                StatementType::AsyncFor(f)
            }
        },
        StatementType::AsyncWith(w) => {
            let binds = w.items.iter().any(|item| {
                item.optional_vars
                    .as_ref()
                    .is_some_and(|v| expr_binds_name(v, from))
            });
            if binds {
                stmt.statement.clone()
            } else {
                let mut w = w.clone();
                w.items = w
                    .items
                    .iter()
                    .map(|item| WithItem {
                        context_expr: rename_expr(&item.context_expr, from, to),
                        optional_vars: item
                            .optional_vars
                            .as_ref()
                            .map(|v| rename_expr(v, from, to)),
                    })
                    .collect();
                w.body = rename_receiver_in_body(&w.body, from, to)?;
                StatementType::AsyncWith(w)
            }
        },
        // Nested scopes: a def/lambda/class that BINDS the target keeps
        // its scope; a nested def WITHOUT such a binding captures the
        // enclosing receiver and must be renamed through. Class bodies are
        // skipped wholesale (their methods have receivers of their own).
        StatementType::FunctionDef(f) => {
            if parameter_list_binds(&f.args, from) || f.name == from {
                StatementType::FunctionDef(f.clone())
            } else {
                let mut f = f.clone();
                f.body = rename_receiver_in_body(&f.body, from, to)?;
                StatementType::FunctionDef(f)
            }
        }
        StatementType::AsyncFunctionDef(f) => {
            if parameter_list_binds(&f.args, from) || f.name == from {
                StatementType::AsyncFunctionDef(f.clone())
            } else {
                let mut f = f.clone();
                f.body = rename_receiver_in_body(&f.body, from, to)?;
                StatementType::AsyncFunctionDef(f)
            }
        }
        StatementType::ClassDef(_) => stmt.statement.clone(),
        // Everything else carries no expressions bound to the receiver.
        other => other.clone(),
    };
    Ok(Statement {
        statement,
        lineno: stmt.lineno,
        col_offset: stmt.col_offset,
        end_lineno: stmt.end_lineno,
        end_col_offset: stmt.end_col_offset,
    })
}

fn rename_expr(expr: &ExprType, from: &str, to: &str) -> ExprType {
    match expr {
        ExprType::Name(n) if n.id == from => ExprType::Name(super::name::Name {
            id: to.to_string(),
            ..n.clone()
        }),
        ExprType::Attribute(a) => ExprType::Attribute(Attribute {
            value: Box::new(rename_expr(a.value.as_ref(), from, to)),
            attr: a.attr.clone(),
            ctx: a.ctx.clone(),
        }),
        ExprType::Subscript(s) => ExprType::Subscript(Subscript {
            value: Box::new(rename_expr(s.value.as_ref(), from, to)),
            kind: rename_subscript_kind(&s.kind, from, to),
            lineno: s.lineno,
            col_offset: s.col_offset,
            end_lineno: s.end_lineno,
            end_col_offset: s.end_col_offset,
        }),
        ExprType::Call(c) => ExprType::Call(Call {
            func: Box::new(rename_expr(c.func.as_ref(), from, to)),
            args: c.args.iter().map(|a| rename_expr(a, from, to)).collect(),
            keywords: c
                .keywords
                .iter()
                .map(|k| Keyword {
                    arg: k.arg.clone(),
                    value: rename_expr(&k.value, from, to),
                    lineno: k.lineno,
                    col_offset: k.col_offset,
                    end_lineno: k.end_lineno,
                    end_col_offset: k.end_col_offset,
                })
                .collect(),
        }),
        ExprType::BoolOp(b) => ExprType::BoolOp(BoolOp {
            op: b.op.clone(),
            values: b.values.iter().map(|v| rename_expr(v, from, to)).collect(),
        }),
        ExprType::BinOp(b) => ExprType::BinOp(super::bin_ops::BinOp {
            op: b.op.clone(),
            left: Box::new(rename_expr(b.left.as_ref(), from, to)),
            right: Box::new(rename_expr(b.right.as_ref(), from, to)),
        }),
        ExprType::UnaryOp(u) => ExprType::UnaryOp(UnaryOp {
            op: u.op.clone(),
            operand: Box::new(rename_expr(u.operand.as_ref(), from, to)),
        }),
        ExprType::Compare(c) => ExprType::Compare(Compare {
            ops: c.ops.clone(),
            left: Box::new(rename_expr(c.left.as_ref(), from, to)),
            comparators: c.comparators.iter().map(|v| rename_expr(v, from, to)).collect(),
        }),
        ExprType::IfExp(i) => ExprType::IfExp(IfExp {
            test: Box::new(rename_expr(i.test.as_ref(), from, to)),
            body: Box::new(rename_expr(i.body.as_ref(), from, to)),
            orelse: Box::new(rename_expr(i.orelse.as_ref(), from, to)),
            lineno: i.lineno,
            col_offset: i.col_offset,
            end_lineno: i.end_lineno,
            end_col_offset: i.end_col_offset,
        }),
        ExprType::NamedExpr(ne) => ExprType::NamedExpr(NamedExpr {
            left: Box::new(rename_expr(ne.left.as_ref(), from, to)),
            right: Box::new(rename_expr(ne.right.as_ref(), from, to)),
        }),
        ExprType::Starred(s) => ExprType::Starred(Starred {
            value: Box::new(rename_expr(s.value.as_ref(), from, to)),
            ctx: s.ctx.clone(),
            lineno: s.lineno,
            col_offset: s.col_offset,
            end_lineno: s.end_lineno,
            end_col_offset: s.end_col_offset,
        }),
        ExprType::List(elts) => {
            ExprType::List(elts.iter().map(|e| rename_expr(e, from, to)).collect())
        }
        ExprType::Tuple(t) => ExprType::Tuple(super::tuple::Tuple {
            elts: t.elts.iter().map(|e| rename_expr(e, from, to)).collect(),
            lineno: t.lineno,
            col_offset: t.col_offset,
            end_lineno: t.end_lineno,
            end_col_offset: t.end_col_offset,
        }),
        ExprType::Set(s) => ExprType::Set(Set {
            elts: s.elts.iter().map(|e| rename_expr(e, from, to)).collect(),
            lineno: s.lineno,
            col_offset: s.col_offset,
            end_lineno: s.end_lineno,
            end_col_offset: s.end_col_offset,
        }),
        ExprType::Dict(d) => ExprType::Dict(super::dict::Dict {
            keys: d.keys.iter().map(|k| k.as_ref().map(|kk| rename_expr(kk, from, to))).collect(),
            values: d.values.iter().map(|v| rename_expr(v, from, to)).collect(),
            lineno: d.lineno,
            col_offset: d.col_offset,
            end_lineno: d.end_lineno,
            end_col_offset: d.end_col_offset,
        }),
        ExprType::JoinedStr(j) => ExprType::JoinedStr(JoinedStr {
            values: j.values.iter().map(|v| rename_expr(v, from, to)).collect(),
            lineno: j.lineno,
            col_offset: j.col_offset,
            end_lineno: j.end_lineno,
            end_col_offset: j.end_col_offset,
        }),
        ExprType::FormattedValue(fv) => {
            ExprType::FormattedValue(FormattedValue {
                value: Box::new(rename_expr(fv.value.as_ref(), from, to)),
                conversion: fv.conversion,
                format_spec: fv
                    .format_spec
                    .as_ref()
                    .map(|s| Box::new(rename_expr(s.as_ref(), from, to))),
                lineno: fv.lineno,
                col_offset: fv.col_offset,
                end_lineno: fv.end_lineno,
                end_col_offset: fv.end_col_offset,
            })
        }
        ExprType::Lambda(l) => {
            // A lambda binding the target name shadows it for its body.
            if parameter_list_binds(&l.args, from) {
                expr.clone()
            } else {
                ExprType::Lambda(Lambda {
                    args: l.args.clone(),
                    body: Box::new(rename_expr(l.body.as_ref(), from, to)),
                    lineno: l.lineno,
                    col_offset: l.col_offset,
                    end_lineno: l.end_lineno,
                    end_col_offset: l.end_col_offset,
                })
            }
        }
        ExprType::ListComp(lc) => ExprType::ListComp(ListComp {
            elt: Box::new(rename_comprehension_elt(&lc.elt, &lc.generators, from, to)),
            generators: renamed_generators(&lc.generators, from, to),
            lineno: lc.lineno,
            col_offset: lc.col_offset,
            end_lineno: lc.end_lineno,
            end_col_offset: lc.end_col_offset,
        }),
        ExprType::SetComp(sc) => ExprType::SetComp(SetComp {
            elt: Box::new(rename_comprehension_elt(&sc.elt, &sc.generators, from, to)),
            generators: renamed_generators(&sc.generators, from, to),
            lineno: sc.lineno,
            col_offset: sc.col_offset,
            end_lineno: sc.end_lineno,
            end_col_offset: sc.end_col_offset,
        }),
        ExprType::DictComp(dc) => ExprType::DictComp(DictComp {
            key: Box::new(rename_comprehension_elt(&dc.key, &dc.generators, from, to)),
            value: Box::new(rename_comprehension_elt(&dc.value, &dc.generators, from, to)),
            generators: renamed_generators(&dc.generators, from, to),
            lineno: dc.lineno,
            col_offset: dc.col_offset,
            end_lineno: dc.end_lineno,
            end_col_offset: dc.end_col_offset,
        }),
        ExprType::GeneratorExp(ge) => ExprType::GeneratorExp(GeneratorExp {
            elt: Box::new(rename_comprehension_elt(&ge.elt, &ge.generators, from, to)),
            generators: renamed_generators(&ge.generators, from, to),
            lineno: ge.lineno,
            col_offset: ge.col_offset,
            end_lineno: ge.end_lineno,
            end_col_offset: ge.end_col_offset,
        }),
        ExprType::Yield(y) => ExprType::Yield(super::yield_expr::Yield {
            value: y.value.as_ref().map(|v| Box::new(rename_expr(v.as_ref(), from, to))),
            lineno: y.lineno,
            col_offset: y.col_offset,
            end_lineno: y.end_lineno,
            end_col_offset: y.end_col_offset,
        }),
        ExprType::YieldFrom(yf) => ExprType::YieldFrom(super::yield_expr::YieldFrom {
            value: Box::new(rename_expr(yf.value.as_ref(), from, to)),
            lineno: yf.lineno,
            col_offset: yf.col_offset,
            end_lineno: yf.end_lineno,
            end_col_offset: yf.end_col_offset,
        }),
        // Constants carry no names.
        other => other.clone(),
    }
}

/// Rename inside a comprehension body only when no generator target binds
/// the target name (a bound name shadows the enclosing receiver there).
fn rename_comprehension_elt(
    elt: &ExprType,
    generators: &[super::list_comp::Comprehension],
    from: &str,
    to: &str,
) -> ExprType {
    let shadowed = generators
        .iter()
        .any(|g| expr_binds_name(&g.target, from));
    if shadowed {
        elt.clone()
    } else {
        rename_expr(elt, from, to)
    }
}

fn renamed_generators(
    generators: &[super::list_comp::Comprehension],
    from: &str,
    to: &str,
) -> Vec<super::list_comp::Comprehension> {
    generators
        .iter()
        .map(|g| super::list_comp::Comprehension {
            target: g.target.clone(),
            iter: rename_expr(&g.iter, from, to),
            ifs: g.ifs.iter().map(|i| rename_expr(i, from, to)).collect(),
            is_async: g.is_async,
        })
        .collect()
}

/// Whether any parameter in the list binds `name` (including *args and
/// **kwargs slots).
fn parameter_list_binds(args: &ParameterList, name: &str) -> bool {
    let named = |p: &arguments::Parameter| p.arg == name;
    args.posonlyargs.iter().any(named)
        || args.args.iter().any(named)
        || args.vararg.as_ref().is_some_and(|p| p.arg == name)
        || args.kwonlyargs.iter().any(named)
        || args.kwarg.as_ref().is_some_and(|p| p.arg == name)
}

/// Whether an expression is (or contains, for tuple/list targets) a
/// binding of `name`.
fn expr_binds_name(expr: &ExprType, name: &str) -> bool {
    match expr {
        ExprType::Name(n) => n.id == name,
        ExprType::Tuple(t) => t.elts.iter().any(|e| expr_binds_name(e, name)),
        ExprType::List(elts) => elts.iter().any(|e| expr_binds_name(e, name)),
        ExprType::Starred(s) => expr_binds_name(s.value.as_ref(), name),
        _ => false,
    }
}

/// A subscript's kind may embed index/slice expressions carrying names.
fn rename_subscript_kind(
    kind: &super::subscript::SubscriptKind,
    from: &str,
    to: &str,
) -> super::subscript::SubscriptKind {
    match kind {
        super::subscript::SubscriptKind::Index(i) => {
            super::subscript::SubscriptKind::Index(Box::new(rename_expr(i.as_ref(), from, to)))
        }
        other => other.clone(),
    }
}

