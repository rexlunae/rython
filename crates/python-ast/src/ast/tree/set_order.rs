//! Set iteration order (issue #444).
//!
//! A Python set lowers to `std::collections::HashSet`, whose iteration
//! order is Rust's (randomized per process), not CPython's table order.
//! Anything that observes the order a set yields its elements in would
//! silently print or compute something different from CPython, and
//! differently from run to run: a `for` loop, `list(s)`, `", ".join(s)`, a
//! list comprehension, unpacking. Those are conversion errors naming
//! `sorted(s)` as the rewrite.
//!
//! What stays allowed is every consumer whose result cannot depend on the
//! order: `len`, `in`, the set algebra and set methods, `set(s)` /
//! `frozenset(s)`, `any` / `all`, `sorted` / `min` / `max` without a
//! `key=` (a key can tie distinct elements, and the tie keeps the
//! iteration order), `sum` over integers, a set comprehension, and a
//! comprehension handed straight to one of those consumers when its body
//! cannot run code that observes the order. A set passed to a user
//! function or method is the callee's business: its own body is checked
//! when it is lowered.

use crate::ast::tree::visit::{self, Descend, Flow};
use crate::{
    CodeGenContext, Comprehension, ExprType, PythonOptions, Statement, StatementType,
    SymbolTableScopes, TypeInfo,
};

/// Builtins whose result does not depend on the order of the iterable
/// they consume (with the per-builtin conditions in [`order_free_call`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderFreeBuiltin {
    Len,
    Set,
    Frozenset,
    Any,
    All,
    Sorted,
    Min,
    Max,
    Sum,
    Isinstance,
    Bool,
}

impl OrderFreeBuiltin {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "len" => Self::Len,
            "set" => Self::Set,
            "frozenset" => Self::Frozenset,
            "any" => Self::Any,
            "all" => Self::All,
            "sorted" => Self::Sorted,
            "min" => Self::Min,
            "max" => Self::Max,
            "sum" => Self::Sum,
            "isinstance" => Self::Isinstance,
            "bool" => Self::Bool,
            _ => return None,
        })
    }
}

/// Builtins a comprehension body may call without running code that
/// could observe the iteration order (no user code, no mutation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PureBuiltin {
    Len,
    Str,
    Int,
    Float,
    Abs,
    Ord,
    Chr,
    Bool,
    Round,
    Min,
    Max,
    Sorted,
    Sum,
    Any,
    All,
    Tuple,
}

impl PureBuiltin {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "len" => Self::Len,
            "str" => Self::Str,
            "int" => Self::Int,
            "float" => Self::Float,
            "abs" => Self::Abs,
            "ord" => Self::Ord,
            "chr" => Self::Chr,
            "bool" => Self::Bool,
            "round" => Self::Round,
            "min" => Self::Min,
            "max" => Self::Max,
            "sorted" => Self::Sorted,
            "sum" => Self::Sum,
            "any" => Self::Any,
            "all" => Self::All,
            "tuple" => Self::Tuple,
            _ => return None,
        })
    }
}

/// The builtins that copy an iterable into a sequence in its iteration
/// order: harmless when an order-free consumer takes the copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderedCopy {
    List,
    Tuple,
}

impl OrderedCopy {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "list" => Some(Self::List),
            "tuple" => Some(Self::Tuple),
            _ => None,
        }
    }
}

struct Scope<'a> {
    ctx: &'a CodeGenContext,
    options: &'a PythonOptions,
    symbols: &'a SymbolTableScopes,
}

impl Scope<'_> {
    fn type_of(&self, e: &ExprType) -> TypeInfo {
        let mut t = crate::infer_type(Some(self.ctx), e, self.options, self.symbols);
        loop {
            t = match t {
                TypeInfo::Borrowed(inner) | TypeInfo::Option(inner) => *inner,
                t => return t,
            };
        }
    }

    fn is_set(&self, e: &ExprType) -> bool {
        // A display or comprehension is visibly a set; anything else asks
        // the inferrer.
        match e {
            ExprType::Set(_) | ExprType::SetComp(_) => true,
            ExprType::Constant(_) | ExprType::List(_) | ExprType::Tuple(_) | ExprType::Dict(_) => {
                false
            }
            _ => matches!(self.type_of(e), TypeInfo::HashSet(_)),
        }
    }

    /// Whether `name` is the builtin, not a binding of this module or
    /// function that shadows it.
    fn is_builtin(&self, name: &str) -> bool {
        self.symbols.get(name).is_none()
            && !self.options.called_params.contains(name)
            && !self.options.param_type_vars.contains_key(name)
            && !self.options.local_types.contains_key(name)
            && !self.options.name_types.contains_key(name)
    }

    /// The element type a comprehension's generators bind to each simple
    /// target name (`x` in `for x in s` over a `set[str]`): the scope's
    /// name types do not know the comprehension's own variables.
    fn targets(&self, generators: &[Comprehension]) -> Vec<(String, TypeInfo)> {
        generators
            .iter()
            .filter_map(|g| {
                let ExprType::Name(n) = &g.target else {
                    return None;
                };
                match self.type_of(&g.iter) {
                    TypeInfo::HashSet(elt) | TypeInfo::Vec(elt) => Some((n.id.clone(), *elt)),
                    _ => None,
                }
            })
            .collect()
    }

    /// The type of `e` inside a comprehension binding `targets`.
    fn type_in(&self, e: &ExprType, targets: &[(String, TypeInfo)]) -> TypeInfo {
        if let ExprType::Name(n) = e
            && let Some((_, t)) = targets.iter().rev().find(|(name, _)| *name == n.id)
        {
            return t.clone();
        }
        self.type_of(e)
    }

    /// Whether a receiver's type is an immutable builtin value whose
    /// methods run no user code and mutate nothing.
    fn immutable_value(&self, e: &ExprType, targets: &[(String, TypeInfo)]) -> bool {
        matches!(
            self.type_in(e, targets),
            TypeInfo::String
                | TypeInfo::StrRef
                | TypeInfo::Bytes
                | TypeInfo::Int
                | TypeInfo::Float
                | TypeInfo::Bool
                | TypeInfo::PyTuple(_)
                | TypeInfo::Tuple(_)
        )
    }

    /// Whether a builtin container or string receiver (`out.extend(s)`,
    /// `", ".join(s)`, `dict.fromkeys(s)`) whose method consumes an
    /// iterable argument in order.
    fn builtin_receiver(&self, e: &ExprType) -> bool {
        if let ExprType::Name(n) = e
            && matches!(n.id.as_str(), "dict" | "list" | "str" | "bytes" | "tuple")
            && self.is_builtin(&n.id)
        {
            return true;
        }
        matches!(
            self.type_of(e),
            TypeInfo::String
                | TypeInfo::StrRef
                | TypeInfo::Bytes
                | TypeInfo::Vec(_)
                | TypeInfo::Dict(..)
                | TypeInfo::PyTuple(_)
                | TypeInfo::Collection(..)
        )
    }

    /// Whether evaluating `e` (a comprehension's element or condition)
    /// cannot run code that observes the order it is evaluated in: no
    /// user call, no mutation, no walrus, no yield.
    fn pure(&self, e: &ExprType, targets: &[(String, TypeInfo)]) -> bool {
        !visit::any_expr(e, |sub| match sub {
            ExprType::Call(c) => match c.func.as_ref() {
                ExprType::Name(n) => {
                    !(PureBuiltin::from_name(&n.id).is_some() && self.is_builtin(&n.id))
                }
                ExprType::Attribute(a) => !self.immutable_value(&a.value, targets),
                _ => true,
            },
            ExprType::NamedExpr(_)
            | ExprType::Yield(_)
            | ExprType::YieldFrom(_)
            | ExprType::Await(_)
            | ExprType::Lambda(_) => true,
            _ => false,
        })
    }

    fn pure_comprehension(&self, elts: &[&ExprType], generators: &[Comprehension]) -> bool {
        let targets = self.targets(generators);
        elts.iter().all(|e| self.pure(e, &targets))
            && generators.iter().all(|g| g.ifs.iter().all(|i| self.pure(i, &targets)))
    }

    /// Whether the elements an iterable yields are integers (`sum` over
    /// integers is exact in any order; over floats it is not).
    fn int_elements(&self, e: &ExprType) -> bool {
        let int = |t: &TypeInfo| matches!(t, TypeInfo::Int | TypeInfo::Bool);
        if let ExprType::GeneratorExp(g) = e {
            return int(&self.type_in(&g.elt, &self.targets(&g.generators)));
        }
        matches!(
            self.type_of(e),
            TypeInfo::HashSet(elt) | TypeInfo::Vec(elt) if int(&elt)
        )
    }

    /// Whether `call` is an order-free consumer of its iterable argument.
    fn order_free_call(&self, call: &crate::Call) -> bool {
        let ExprType::Name(n) = call.func.as_ref() else {
            return false;
        };
        let Some(builtin) = OrderFreeBuiltin::from_name(&n.id) else {
            return false;
        };
        if !self.is_builtin(&n.id) {
            return false;
        }
        let keyed = call.keywords.iter().any(|k| k.arg.as_deref() == Some("key"));
        match builtin {
            OrderFreeBuiltin::Sorted | OrderFreeBuiltin::Min | OrderFreeBuiltin::Max => !keyed,
            OrderFreeBuiltin::Sum => call.args.first().is_some_and(|a| self.int_elements(a)),
            _ => true,
        }
    }
}

/// The comprehension's element expressions and generators, when `e` is one.
fn comprehension(e: &ExprType) -> Option<(Vec<&ExprType>, &[Comprehension])> {
    match e {
        ExprType::ListComp(c) => Some((vec![c.elt.as_ref()], &c.generators)),
        ExprType::GeneratorExp(c) => Some((vec![c.elt.as_ref()], &c.generators)),
        ExprType::SetComp(c) => Some((vec![c.elt.as_ref()], &c.generators)),
        ExprType::DictComp(c) => Some((vec![c.key.as_ref(), c.value.as_ref()], &c.generators)),
        _ => None,
    }
}

fn refusal(what: &str, line: Option<usize>) -> String {
    let at = line.map(|l| format!(" at line {}", l)).unwrap_or_default();
    format!(
        "{}{} is not supported yet: a set lowers to a Rust HashSet, whose \
         iteration order differs from CPython's (and from run to run), so \
         the result would come out in a different order; iterate \
         `sorted(s)` instead (or another order-independent consumer: len, \
         in, set operations, any/all, min/max/sorted without key=, sum of \
         ints) — rython refuses to silently reorder it (issue #444)",
        what, at
    )
}

/// Refuse every order-observable iteration of a set in `body` (the
/// statements of one scope; nested definitions are checked when they are
/// lowered).
pub(crate) fn check_body(
    body: &[Statement],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<(), String> {
    let scope = Scope { ctx, options, symbols };
    let mut err: Option<String> = None;
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        match check_stmt(&scope, s) {
            Ok(()) => Flow::Continue,
            Err(e) => {
                err = Some(e);
                Flow::Stop
            }
        }
    });
    err.map_or(Ok(()), Err)
}

fn check_stmt(scope: &Scope<'_>, s: &Statement) -> Result<(), String> {
    let line = s.lineno;
    match &s.statement {
        StatementType::For(f) if scope.is_set(&f.iter) => {
            return Err(refusal("a `for` loop over a set", line));
        }
        StatementType::AsyncFor(f) if scope.is_set(&f.iter) => {
            return Err(refusal("an `async for` loop over a set", line));
        }
        StatementType::Assign(a)
            if a.targets.iter().any(|t| matches!(t, ExprType::Tuple(_) | ExprType::List(_)))
                && scope.is_set(&a.value) =>
        {
            return Err(refusal("unpacking a set into names", line));
        }
        _ => {}
    }
    // Comprehensions handed straight to an order-free consumer, recorded
    // as their parent call is met (the walk is pre-order).
    let mut allowed: Vec<*const ExprType> = Vec::new();
    for e in visit::stmt_exprs(s) {
        let mut found: Option<&'static str> = None;
        visit::walk_expr(e, &mut |sub| {
            if found.is_some() {
                return;
            }
            found = check_expr(scope, sub, &mut allowed);
        });
        if let Some(what) = found {
            return Err(refusal(what, line));
        }
    }
    Ok(())
}

/// Record `arg`, the sole argument of an order-free consumer, as
/// allowed to iterate a set: a pure comprehension, or a `list(...)` /
/// `tuple(...)` copy of the set whose order the consumer then ignores
/// (`sorted(list({r for r in ranges if r}))` — charset_normalizer's
/// alphabets), together with that copy's own pure comprehension.
fn allow_order_free_arg(scope: &Scope<'_>, arg: &ExprType, allowed: &mut Vec<*const ExprType>) {
    if let Some((elts, generators)) = comprehension(arg) {
        if scope.pure_comprehension(&elts, generators) {
            allowed.push(arg as *const ExprType);
        }
        return;
    }
    if let ExprType::Call(copy) = arg
        && let ExprType::Name(n) = copy.func.as_ref()
        && OrderedCopy::from_name(&n.id).is_some()
        && scope.is_builtin(&n.id)
        && copy.keywords.is_empty()
        && let [inner] = copy.args.as_slice()
    {
        allowed.push(arg as *const ExprType);
        allow_order_free_arg(scope, inner, allowed);
    }
}

fn check_expr(
    scope: &Scope<'_>,
    e: &ExprType,
    allowed: &mut Vec<*const ExprType>,
) -> Option<&'static str> {
    let starred_set = |elts: &[ExprType]| {
        elts.iter()
            .any(|x| matches!(x, ExprType::Starred(st) if scope.is_set(&st.value)))
    };
    match e {
        ExprType::Call(call) => {
            if starred_set(&call.args) {
                return Some("unpacking a set into call arguments (`f(*s)`)");
            }
            if scope.order_free_call(call) {
                if let [arg] = call.args.as_slice() {
                    allow_order_free_arg(scope, arg, allowed);
                }
                return None;
            }
            if allowed.contains(&(e as *const ExprType)) {
                return None;
            }
            let set_arg = call.args.iter().any(|a| scope.is_set(a))
                || call.keywords.iter().any(|k| scope.is_set(&k.value));
            match call.func.as_ref() {
                ExprType::Name(n) if scope.is_builtin(&n.id) => {
                    if set_arg {
                        return Some(match n.id.as_str() {
                            "list" => "`list(s)` of a set",
                            "tuple" => "`tuple(s)` of a set",
                            "enumerate" => "`enumerate(s)` of a set",
                            "zip" => "`zip(...)` over a set",
                            "iter" => "`iter(s)` of a set",
                            "print" => "printing a set",
                            "str" | "repr" => "the string form of a set",
                            _ => "passing a set to an order-observing builtin",
                        });
                    }
                }
                // A set receiver's own methods are order-free (`s.pop()`
                // is checked at run time: `py_set_pop`).
                ExprType::Attribute(a) => {
                    if set_arg && !scope.is_set(&a.value) && scope.builtin_receiver(&a.value) {
                        return Some(match a.attr.as_str() {
                            "join" => "`str.join` over a set",
                            "extend" => "`list.extend` from a set",
                            "fromkeys" => "`dict.fromkeys` over a set",
                            _ => "passing a set to an order-observing method",
                        });
                    }
                }
                _ => {}
            }
            None
        }
        ExprType::List(elts) if starred_set(elts) => Some("unpacking a set into a list (`[*s]`)"),
        ExprType::Tuple(t) if starred_set(&t.elts) => Some("unpacking a set into a tuple (`(*s,)`)"),
        ExprType::YieldFrom(y) if scope.is_set(&y.value) => Some("`yield from` a set"),
        _ => {
            let (elts, generators) = comprehension(e)?;
            if !generators.iter().any(|g| scope.is_set(&g.iter)) {
                return None;
            }
            if allowed.contains(&(e as *const ExprType)) {
                return None;
            }
            if matches!(e, ExprType::SetComp(_)) && scope.pure_comprehension(&elts, generators) {
                return None;
            }
            Some(match e {
                ExprType::ListComp(_) => "a list comprehension over a set",
                ExprType::DictComp(_) => "a dict comprehension over a set",
                ExprType::SetComp(_) => "a set comprehension over a set whose body calls code",
                _ => "a generator expression over a set",
            })
        }
    }
}
