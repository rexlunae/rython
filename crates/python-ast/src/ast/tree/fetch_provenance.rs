//! The fetch-provenance analysis (issue #137, Directive 4's borrowed-
//! accessor increment): a local assigned from a CONTAINER fetch
//! (`item = self.items.get(k)`, `item = self.items[k]`, or a one-hop
//! method whose body is a single such return — `find`) is a VIEW of the
//! container slot in CPython (the local holds a reference to the stored
//! object). Its mutations must reach the slot — the write-back model
//! lowers `item.qty -= qty` to: mutate a copy, store it back into the
//! slot, and rebind the local to the mutated value. Reads of the local
//! stay the clone (a strong reference — CPython-faithful), so only the
//! mutation sites change.

use crate::{
    CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    ast::tree::statement::StatementType,
};

/// The provenance of a fetch-local: the CONTAINER expression (a
/// `self.<field>` path) and the KEY expression of the fetch that bound
/// it. `None` when the binding is not a resolvable container fetch.
#[derive(Debug, Clone)]
pub struct FetchProvenance {
    pub container: ExprType,
    pub key: ExprType,
}

/// Resolve the provenance of a local assigned from a container fetch.
/// Two shapes:
/// 1. DIRECT: `item = self.<field>.get(k)` / `self.<field>[k]`.
/// 2. ONE-HOP: `item = self.<method>(k)` whose body is a single `return`
///    of a direct fetch — the method's parameters substitute the call's
///    arguments POSITIONALLY, and only when every substituted argument is
///    a simple expression (a name or a literal — complex argument
///    expressions refuse loudly rather than guess).
pub fn fetch_provenance(
    local: &str,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<FetchProvenance> {
    let crate::SymbolTableNode::Assign { value, .. } = symbols.get(local)? else {
        return None;
    };
    fetch_provenance_of_expr(value, ctx, options, symbols, 0)
}

fn fetch_provenance_of_expr(
    expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
    depth: u32,
) -> Option<FetchProvenance> {
    if depth > 2 {
        return None;
    }
    let ExprType::Call(call) = expr else {
        // A SUBSCRIPT fetch: `self.<field>[k]` — no Option (the direct
        // read), but the same slot.
        if let ExprType::Subscript(sub) = expr {
            let (field, _class) = container_field(&sub.value, ctx, options, symbols)?;
            let crate::SubscriptKind::Index(key) = &sub.kind else {
                return None;
            };
            return Some(FetchProvenance {
                container: field,
                key: (**key).clone(),
            });
        }
        return None;
    };
    let ExprType::Attribute(attr) = call.func.as_ref() else {
        return None;
    };
    // DIRECT: `self.<field>.get(k)` — the receiver is the container
    // FIELD (a `self.<field>` path; the container lives on the same
    // object, so the slot is writable).
    if attr.attr == "get" {
        if let Some((field, _class)) = container_field(&attr.value, ctx, options, symbols) {
            let key = call.args.first()?.clone();
            return Some(FetchProvenance {
                container: field,
                key,
            });
        }
    }
    // ONE-HOP: `self.<method>(args)` — the method's body must be a
    // single `return <direct fetch>`; its parameters substitute the
    // call's arguments positionally (simple arguments only).
    let class_name = ctx.enclosing_class_name()?.to_string();
    let crate::SymbolTableNode::ClassDef(class) = symbols.get(&class_name)? else {
        return None;
    };
    let method = class.method_on_mro(&attr.attr, symbols)?;
    if method.decorator_list.iter().any(|d| {
        matches!(d, ExprType::Name(n) if n.id == "staticmethod" || n.id == "classmethod")
    }) {
        return None;
    }
    let body = method.body.last()?;
    let StatementType::Return(Some(ret)) = &body.statement else {
        return None;
    };
    // Substitute the method's parameters with the call's arguments
    // (positional; simple expressions only).
    let params: Vec<String> = method
        .args
        .posonlyargs
        .iter()
        .chain(method.args.args.iter())
        .filter(|p| p.arg != "self")
        .map(|p| p.arg.clone())
        .collect();
    let mut substituted = ret.value.clone();
    for (i, param) in params.iter().enumerate() {
        let Some(arg) = call.args.get(i) else {
            return None;
        };
        if !is_simple_expr(arg) {
            return None;
        }
        substitute_name(&mut substituted, param, arg);
    }
    fetch_provenance_of_expr(&substituted, ctx, options, symbols, depth + 1)
}

/// The container field a fetch reads through: `self.<field>` where the
/// field's type is a dict (the class table is the authority). Returns the
/// FIELD EXPRESSION (for the write-back's py_set_index receiver).
fn container_field(
    expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<(ExprType, crate::ClassDef)> {
    let ExprType::Attribute(attr) = expr else {
        return None;
    };
    let ExprType::Name(recv) = attr.value.as_ref() else {
        return None;
    };
    if recv.id != "self" {
        return None;
    }
    let class_name = ctx.enclosing_class_name()?.to_string();
    let crate::SymbolTableNode::ClassDef(class) = symbols.get(&class_name)? else {
        return None;
    };
    for c in class.base_chain(symbols) {
        if let Ok(fields) = c.infer_fields(symbols, options)
            && let Some((_, ty)) = fields.iter().find(|(name, _)| *name == attr.attr)
            && matches!(ty, crate::TypeInfo::Dict(_, _))
        {
            return Some((expr.clone(), class.clone()));
        }
    }
    None
}

fn is_simple_expr(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Name(_) | ExprType::Constant(_))
}

/// Replace every free `Name(param)` in `expr` with `arg` (a textual walk —
/// the shapes here are small: the fetch's key chain).
fn substitute_name(expr: &mut ExprType, param: &str, arg: &ExprType) {
    match expr {
        ExprType::Name(n) if n.id == param => *expr = arg.clone(),
        ExprType::Attribute(a) => substitute_name(&mut a.value, param, arg),
        ExprType::Call(c) => {
            substitute_name(&mut c.func, param, arg);
            for a in &mut c.args {
                substitute_name(a, param, arg);
            }
            for kw in &mut c.keywords {
                substitute_name(&mut kw.value, param, arg);
            }
        }
        ExprType::Subscript(s) => {
            substitute_name(&mut s.value, param, arg);
            if let crate::SubscriptKind::Index(k) = &mut s.kind {
                substitute_name(k, param, arg);
            }
        }
        _ => {}
    }
}
