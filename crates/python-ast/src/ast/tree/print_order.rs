//! Evaluation order of `print`'s arguments (issue #433).
//!
//! CPython evaluates EVERY argument of `print(a, b, ...)` first and only
//! then converts each with `str()`, so a later argument that mutates an
//! object an earlier one names is visible in the earlier one's output:
//! `print(xs, xs.pop())` shows the list AFTER the pop. The lowering
//! renders each argument to a string as soon as it is evaluated, which
//! shows the stale state.
//!
//! The fix defers the rendering of exactly the arguments for which
//! deferral is observably equal to CPython: a PURE PLACE READ (a name, or
//! an attribute chain) of a MUTABLE container or class instance, followed
//! by a later argument that can run code. A place read has no side effect,
//! so evaluating it late is the same as evaluating it early, and the later
//! rendering sees whatever the later arguments did to the object. Every
//! other argument evaluates eagerly in source order, as before — a slice,
//! a `len(xs)` or a scalar field read is correctly a snapshot — and when no
//! later argument can run code the generated shape is untouched.
//!
//! Two situations cannot be deferred faithfully and are conversion errors
//! rather than silent divergences:
//! - a later argument REBINDS the place (a walrus on its root name, or a
//!   called method that assigns the attribute): CPython has already
//!   captured the old object, so neither the eager nor the deferred
//!   rendering is right in general;
//! - a subscript of a mutable container (`print(grid[0], grid[0].pop())`),
//!   whose fallible read (IndexError / KeyError) must happen at its own
//!   position, not after the later arguments' side effects.

use crate::ast::tree::scope::mutates_receiver;
use crate::ast::tree::visit::{self, Descend};
use crate::{
    CodeGenContext, ExprType, PythonOptions, Statement, StatementType, SymbolTableNode,
    SymbolTableScopes, TypeInfo,
};

/// Builtins that only READ their arguments: a call to one of these (with
/// no callback among its arguments) cannot mutate anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadOnlyBuiltin {
    Abs,
    All,
    Any,
    Ascii,
    Bin,
    Bool,
    Bytes,
    Callable,
    Chr,
    Divmod,
    Enumerate,
    Float,
    Format,
    Frozenset,
    Getattr,
    Hasattr,
    Hash,
    Hex,
    Id,
    Int,
    Isinstance,
    Issubclass,
    Len,
    List,
    Max,
    Min,
    Oct,
    Ord,
    Pow,
    Range,
    Repr,
    Reversed,
    Round,
    Set,
    Sorted,
    Str,
    Sum,
    Tuple,
    Type,
    Zip,
}

impl ReadOnlyBuiltin {
    /// The one string match: a builtin's name to its classification.
    fn from_name(name: &str) -> Option<ReadOnlyBuiltin> {
        use ReadOnlyBuiltin as B;
        Some(match name {
            "abs" => B::Abs,
            "all" => B::All,
            "any" => B::Any,
            "ascii" => B::Ascii,
            "bin" => B::Bin,
            "bool" => B::Bool,
            "bytes" => B::Bytes,
            "callable" => B::Callable,
            "chr" => B::Chr,
            "divmod" => B::Divmod,
            "enumerate" => B::Enumerate,
            "float" => B::Float,
            "format" => B::Format,
            "frozenset" => B::Frozenset,
            "getattr" => B::Getattr,
            "hasattr" => B::Hasattr,
            "hash" => B::Hash,
            "hex" => B::Hex,
            "id" => B::Id,
            "int" => B::Int,
            "isinstance" => B::Isinstance,
            "issubclass" => B::Issubclass,
            "len" => B::Len,
            "list" => B::List,
            "max" => B::Max,
            "min" => B::Min,
            "oct" => B::Oct,
            "ord" => B::Ord,
            "pow" => B::Pow,
            "range" => B::Range,
            "repr" => B::Repr,
            "reversed" => B::Reversed,
            "round" => B::Round,
            "set" => B::Set,
            "sorted" => B::Sorted,
            "str" => B::Str,
            "sum" => B::Sum,
            "tuple" => B::Tuple,
            "type" => B::Type,
            "zip" => B::Zip,
            _ => return None,
        })
    }
}

/// A pure place read: what can be rendered late without changing what the
/// program observes.
enum Place {
    /// A bare name.
    Name(String),
    /// An attribute chain `root.a.b`; `attr` is the last attribute.
    Attr { root: String, attr: String },
    /// A subscript of a place (`grid[0]`).
    Index { root: String },
}

impl Place {
    fn root(&self) -> &str {
        match self {
            Place::Name(r) | Place::Attr { root: r, .. } | Place::Index { root: r } => r,
        }
    }
}

/// The root name of a name / attribute / subscript chain.
fn chain_root(e: &ExprType) -> Option<String> {
    match e {
        ExprType::Name(n) => Some(n.id.clone()),
        ExprType::Attribute(a) => chain_root(&a.value),
        ExprType::Subscript(s) => chain_root(&s.value),
        _ => None,
    }
}

/// The source spelling of a place, for messages.
fn place_text(e: &ExprType) -> String {
    match e {
        ExprType::Name(n) => n.id.clone(),
        ExprType::Attribute(a) => format!("{}.{}", place_text(&a.value), a.attr),
        ExprType::Subscript(s) => format!("{}[...]", place_text(&s.value)),
        _ => "<expression>".to_string(),
    }
}

/// Whether evaluating `e` can run arbitrary code (so it may mutate or
/// rebind something an earlier argument names): a call that is not
/// provably read-only, an await / yield, or a walrus.
fn may_run_code(
    e: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    visit::any_expr(e, |sub| match sub {
        ExprType::Call(c) => !call_is_read_only(c, ctx, options, symbols),
        ExprType::Await(_)
        | ExprType::Yield(_)
        | ExprType::YieldFrom(_)
        | ExprType::NamedExpr(_) => true,
        _ => false,
    })
}

/// A callback among a call's arguments (a lambda, a user function by
/// name) makes even a read-only builtin run user code.
fn is_callback(e: &ExprType, symbols: &SymbolTableScopes) -> bool {
    match e {
        ExprType::Lambda(_) => true,
        ExprType::Name(n) => matches!(
            symbols.get(&n.id),
            Some(SymbolTableNode::FunctionDef(_)) | Some(SymbolTableNode::ClassDef(_))
        ),
        _ => false,
    }
}

fn call_is_read_only(
    call: &crate::Call,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    if call.args.iter().any(|a| is_callback(a, symbols))
        || call.keywords.iter().any(|k| is_callback(&k.value, symbols))
    {
        return false;
    }
    match call.func.as_ref() {
        // A builtin spelling not shadowed by a user definition.
        ExprType::Name(n) => symbols.get(&n.id).is_none() && ReadOnlyBuiltin::from_name(&n.id).is_some(),
        // A method on a builtin value: read-only unless it is a known
        // in-place mutator. A user class (or anything unresolved) may run
        // arbitrary code.
        ExprType::Attribute(a) => {
            let recv = crate::infer_type(Some(ctx), &a.value, options, symbols);
            builtin_value_type(&recv) && !mutates_receiver(&a.attr)
        }
        _ => false,
    }
}

/// Whether a receiver type is a builtin value (whose methods are the
/// runtime's, never user code).
fn builtin_value_type(t: &TypeInfo) -> bool {
    match t {
        TypeInfo::Int
        | TypeInfo::Float
        | TypeInfo::Bool
        | TypeInfo::StrRef
        | TypeInfo::String
        | TypeInfo::Bytes
        | TypeInfo::Vec(_)
        | TypeInfo::PyTuple(_)
        | TypeInfo::HashSet(_)
        | TypeInfo::Dict(..)
        | TypeInfo::Tuple(_)
        | TypeInfo::Collection(..)
        | TypeInfo::Complex => true,
        TypeInfo::Borrowed(inner) | TypeInfo::Option(inner) => builtin_value_type(inner),
        _ => false,
    }
}

/// Whether a value of this type is observably mutable through another
/// reference (so its rendering depends on WHEN it is rendered). Only
/// statically known containers and class instances qualify: a scalar or a
/// string is the same object whenever it is rendered, and an
/// unresolved-type name is left to the eager shape.
fn observably_mutable(t: &TypeInfo) -> bool {
    match t {
        TypeInfo::Vec(_)
        | TypeInfo::HashSet(_)
        | TypeInfo::Dict(..)
        | TypeInfo::Collection(..)
        | TypeInfo::NdArray
        | TypeInfo::Class(_) => true,
        TypeInfo::Borrowed(inner) => observably_mutable(inner),
        _ => false,
    }
}

/// Classify an argument as a pure place read, if it is one.
fn place_of(
    e: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<Place> {
    match e {
        ExprType::Name(n) => Some(Place::Name(n.id.clone())),
        ExprType::Attribute(a) => {
            // A property read lowers to a getter CALL (a fresh value), not
            // a place.
            if crate::ast::tree::attribute::attribute_read_is_call(a, ctx, symbols, options) {
                return None;
            }
            let root = chain_root(e)?;
            // Every link must itself be a place (no call in the chain).
            let mut cur = a.value.as_ref();
            loop {
                match cur {
                    ExprType::Name(_) => break,
                    ExprType::Attribute(inner) => cur = inner.value.as_ref(),
                    _ => return None,
                }
            }
            Some(Place::Attr {
                root,
                attr: a.attr.clone(),
            })
        }
        ExprType::Subscript(s) => {
            let root = chain_root(e)?;
            matches!(s.kind, crate::ast::tree::subscript::SubscriptKind::Index(_))
                .then_some(Place::Index { root })
        }
        _ => None,
    }
}

/// The user function or method a call resolves to, with the class it
/// belongs to (for `self.m()` inside its own methods).
fn resolve_callee<'s>(
    call: &crate::Call,
    class: Option<&str>,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &'s SymbolTableScopes,
) -> Option<(crate::FunctionDef, Option<String>)> {
    let method_of = |cn: &str, m: &str| -> Option<(crate::FunctionDef, Option<String>)> {
        let Some(SymbolTableNode::ClassDef(c)) = symbols.get(cn) else {
            return None;
        };
        c.body.iter().find_map(|s| match &s.statement {
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) if f.name == m => {
                Some((f.clone(), Some(cn.to_string())))
            }
            _ => None,
        })
    };
    match call.func.as_ref() {
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::FunctionDef(f)) => Some((f.clone(), None)),
            Some(SymbolTableNode::ClassDef(_)) => method_of(&n.id, "__init__"),
            _ => None,
        },
        ExprType::Attribute(a) => {
            if visit::is_self(&a.value) {
                if let Some(cn) = class {
                    return method_of(cn, &a.attr);
                }
            }
            match crate::infer_type(Some(ctx), &a.value, options, symbols) {
                TypeInfo::Class(cn) => method_of(&cn, &a.attr),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether the body of `f` (or anything it transitively calls among the
/// module's own functions and methods) assigns an attribute named `attr`.
fn body_rebinds_attr(
    f: &crate::FunctionDef,
    class: Option<&str>,
    attr: &str,
    depth: usize,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    // A store TO the attribute itself (directly or inside an unpacking
    // target); `self.items[0] = x` stores into the container, not the
    // attribute.
    fn assigns_attr(t: &ExprType, attr: &str) -> bool {
        match t {
            ExprType::Attribute(a) => a.attr == attr,
            ExprType::Tuple(t) => t.elts.iter().any(|e| assigns_attr(e, attr)),
            ExprType::List(items) => items.iter().any(|e| assigns_attr(e, attr)),
            ExprType::Starred(st) => assigns_attr(&st.value, attr),
            _ => false,
        }
    }
    let is_store = |t: &ExprType| assigns_attr(t, attr);
    let stores = visit::any_stmt(&f.body, Descend::All, |s: &Statement| {
        let plain_targets = match &s.statement {
            // `self.items += [x]` mutates a list in place; it is not a
            // rebinding of a container.
            StatementType::AugAssign(_) => Vec::new(),
            StatementType::Delete(ts) => ts.iter().collect(),
            _ => visit::stmt_targets(s),
        };
        plain_targets.into_iter().any(is_store)
    });
    if stores {
        return true;
    }
    if depth == 0 {
        return false;
    }
    visit::any_stmt(&f.body, Descend::All, |s: &Statement| {
        visit::stmt_all_exprs(s).into_iter().any(|e| {
            visit::any_expr(e, |sub| match sub {
                ExprType::Call(c) => resolve_callee(c, class, ctx, options, symbols).is_some_and(
                    |(callee, cls)| {
                        body_rebinds_attr(
                            &callee,
                            cls.as_deref().or(class),
                            attr,
                            depth - 1,
                            ctx,
                            options,
                            symbols,
                        )
                    },
                ),
                _ => false,
            })
        })
    })
}

/// Which of `args` render AFTER every argument has evaluated (the mask is
/// all-false when the eager shape is already right). `later_extra` are the
/// keyword values (`sep=`, `end=`, `flush=`), evaluated after every
/// positional argument.
pub(crate) fn deferred_renders(
    args: &[ExprType],
    later_extra: &[&ExprType],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    let mut mask = vec![false; args.len()];
    for (i, arg) in args.iter().enumerate() {
        let Some(place) = place_of(arg, ctx, options, symbols) else {
            continue;
        };
        let ty = crate::infer_type(Some(ctx), arg, options, symbols);
        if !observably_mutable(&ty) {
            continue;
        }
        let later: Vec<&ExprType> = args[i + 1..].iter().chain(later_extra.iter().copied()).collect();
        if !later.iter().any(|e| may_run_code(e, ctx, options, symbols)) {
            continue;
        }
        let text = place_text(arg);
        let root = place.root().to_string();

        // A walrus on the root name rebinds it mid-call.
        let rebinds_root = later.iter().any(|e| {
            visit::any_expr(e, |sub| {
                matches!(sub, ExprType::NamedExpr(n)
                    if matches!(n.left.as_ref(), ExprType::Name(t) if t.id == root))
            })
        });
        if rebinds_root {
            return Err(format!(
                "print(): argument `{text}` names a mutable object, and a later argument \
                 rebinds `{root}` with `:=`. CPython evaluates every argument before \
                 converting any of them, so the output depends on which object each \
                 argument captured; rython refuses to silently pick one. Bind the pieces \
                 to separate names first (`a = ...; b = ...; print(a, b)`)."
            )
            .into());
        }

        match &place {
            Place::Name(_) => {}
            Place::Attr { attr, .. } => {
                let rebound = later.iter().any(|e| {
                    visit::any_expr(e, |sub| match sub {
                        ExprType::Call(c) => resolve_callee(c, None, ctx, options, symbols)
                            .is_some_and(|(f, cls)| {
                                body_rebinds_attr(&f, cls.as_deref(), attr, 4, ctx, options, symbols)
                            }),
                        _ => false,
                    })
                });
                if rebound {
                    return Err(format!(
                        "print(): argument `{text}` names a mutable object, and a later \
                         argument calls code that assigns `.{attr}`. CPython evaluates \
                         every argument before converting any of them, so whether the \
                         output shows the old or the new object depends on whether that \
                         call mutates or rebinds it; rython refuses to silently pick one. \
                         Bind the pieces to separate names first (`a = ...; b = ...; \
                         print(a, b)`)."
                    )
                    .into());
                }
            }
            Place::Index { .. } => {
                // The read can raise (IndexError / KeyError) and must do so
                // at its own position, so it cannot be deferred. When a later
                // argument mutates the same root, eager rendering is stale.
                let same_root_mutation = later.iter().any(|e| {
                    visit::any_expr(e, |sub| match sub {
                        ExprType::Call(c) => match c.func.as_ref() {
                            ExprType::Attribute(a) => {
                                chain_root(&a.value).as_deref() == Some(root.as_str())
                                    && !call_is_read_only(c, ctx, options, symbols)
                            }
                            _ => false,
                        },
                        _ => false,
                    })
                });
                if same_root_mutation {
                    return Err(format!(
                        "print(): argument `{text}` is a mutable element, and a later \
                         argument mutates `{root}`. CPython evaluates every argument \
                         before converting any of them, so the element shows its state \
                         AFTER the mutation; rython renders a subscript at its own position \
                         and would silently show the stale state. Bind the element to a \
                         name first (`row = {root}[...]; print(row, ...)`)."
                    )
                    .into());
                }
                continue;
            }
        }
        mask[i] = true;
    }
    Ok(mask)
}

/// Whether a keyword value must be hoisted into a `let` (it can run code,
/// so its position relative to the deferred renders matters).
pub(crate) fn needs_hoist(
    e: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    may_run_code(e, ctx, options, symbols)
}
