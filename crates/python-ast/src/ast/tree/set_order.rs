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
//! iteration order) over elements that are not floats (a NaN compares
//! neither way, so the reduction keeps whichever member came first),
//! `sum` over integers, a set comprehension, and a comprehension handed
//! straight to one of those consumers when no part of it (element,
//! condition, or a later generator's iterable) can run code that
//! observes the order. A comprehension's parts are typed in the scope its
//! preceding generators bind (`for y in x` after `for x in groups` over a
//! `list[set[int]]` iterates a set). A set passed to a user
//! function or method is the callee's business: its own body is checked
//! when it is lowered.

use crate::ast::tree::visit::{self, Descend, Flow};
use crate::{
    CodeGenContext, Comprehension, ExprType, PythonOptions, Statement, StatementType,
    SymbolTableScopes, TypeInfo,
};

/// The builtins this check classifies, parsed once from the call's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Builtin {
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
    Str,
    Repr,
    Int,
    Float,
    Abs,
    Ord,
    Chr,
    Round,
    List,
    Tuple,
    Enumerate,
    Zip,
    Iter,
    Print,
}

impl Builtin {
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
            "str" => Self::Str,
            "repr" => Self::Repr,
            "int" => Self::Int,
            "float" => Self::Float,
            "abs" => Self::Abs,
            "ord" => Self::Ord,
            "chr" => Self::Chr,
            "round" => Self::Round,
            "list" => Self::List,
            "tuple" => Self::Tuple,
            "enumerate" => Self::Enumerate,
            "zip" => Self::Zip,
            "iter" => Self::Iter,
            "print" => Self::Print,
            _ => return None,
        })
    }

    /// Whether the result does not depend on the order of the iterable
    /// it consumes (with the per-builtin conditions in
    /// [`Scope::order_free_call`]).
    fn order_free(self) -> bool {
        matches!(
            self,
            Self::Len
                | Self::Set
                | Self::Frozenset
                | Self::Any
                | Self::All
                | Self::Sorted
                | Self::Min
                | Self::Max
                | Self::Sum
                | Self::Isinstance
                | Self::Bool
        )
    }

    /// Whether a comprehension may call it without running code that
    /// could observe the iteration order (no user code, no mutation).
    fn pure(self) -> bool {
        matches!(
            self,
            Self::Len
                | Self::Str
                | Self::Int
                | Self::Float
                | Self::Abs
                | Self::Ord
                | Self::Chr
                | Self::Bool
                | Self::Round
                | Self::Min
                | Self::Max
                | Self::Sorted
                | Self::Sum
                | Self::Any
                | Self::All
                | Self::Tuple
        )
    }

    /// Whether it copies an iterable into a sequence in its iteration
    /// order: harmless when an order-free consumer takes the copy.
    fn ordered_copy(self) -> bool {
        matches!(self, Self::List | Self::Tuple)
    }

    /// What a refusal calls handing it a set.
    fn refusal(self, keyed: bool) -> &'static str {
        match self {
            Self::List => "`list(s)` of a set",
            Self::Tuple => "`tuple(s)` of a set",
            Self::Enumerate => "`enumerate(s)` of a set",
            Self::Zip => "`zip(...)` over a set",
            Self::Iter => "`iter(s)` of a set",
            Self::Print => "printing a set",
            Self::Str | Self::Repr => "the string form of a set",
            Self::Sorted | Self::Min | Self::Max if keyed => {
                "`sorted`/`min`/`max` with `key=` over a set"
            }
            Self::Sorted | Self::Min | Self::Max => {
                "`sorted`/`min`/`max` over a set of floats (a NaN makes the result depend on the order)"
            }
            Self::Sum => "`sum` over a set of non-integers",
            _ => "passing a set to an order-observing builtin",
        }
    }
}

struct Scope<'a> {
    ctx: &'a CodeGenContext,
    options: &'a PythonOptions,
    symbols: &'a SymbolTableScopes,
}

impl<'a> Scope<'a> {
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
    /// function (or a comprehension target) that shadows it.
    fn is_builtin(&self, name: &str) -> bool {
        self.symbols.get(name).is_none()
            && !self.options.called_params.contains(name)
            && !self.options.param_type_vars.contains_key(name)
            && !self.options.local_types.contains_key(name)
            && !self.options.name_types.contains_key(name)
    }

    /// The builtin `func` names, when it is an unshadowed builtin name.
    fn builtin(&self, func: &ExprType) -> Option<Builtin> {
        let ExprType::Name(n) = func else {
            return None;
        };
        Builtin::from_name(&n.id).filter(|_| self.is_builtin(&n.id))
    }

    /// The same scope with `options` in place of its own (a
    /// comprehension's progressively bound scopes).
    fn with<'b>(&self, options: &'b PythonOptions) -> Scope<'b>
    where
        'a: 'b,
    {
        Scope {
            ctx: self.ctx,
            options,
            symbols: self.symbols,
        }
    }

    /// The typing scopes of a comprehension's parts: entry `i` is the
    /// scope generator `i`'s iterable is evaluated in (the targets of the
    /// generators before it bound), the last one the element's.
    fn prefix_scopes(&self, generators: &[Comprehension]) -> Vec<PythonOptions> {
        crate::ast::tree::type_ctx::comprehension_prefix_scopes(
            generators,
            Some(self.ctx),
            self.options,
            self.symbols,
        )
    }

    /// Whether any generator of a comprehension iterates a set, each
    /// iterable typed in the scope its preceding generators bind.
    fn iterates_set(&self, generators: &[Comprehension]) -> bool {
        let scopes = self.prefix_scopes(generators);
        generators
            .iter()
            .enumerate()
            .any(|(i, g)| self.with(&scopes[i]).is_set(&g.iter))
    }

    /// Whether a receiver's type is an immutable builtin value whose
    /// methods run no user code and mutate nothing.
    fn immutable_value(&self, e: &ExprType) -> bool {
        matches!(
            self.type_of(e),
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

    /// Whether evaluating `e` (a part of a comprehension) cannot run code
    /// that observes the order it is evaluated in: no user call, no
    /// mutation, no walrus, no yield.
    fn pure(&self, e: &ExprType) -> bool {
        !visit::any_expr(e, |sub| match sub {
            ExprType::Call(c) => match c.func.as_ref() {
                ExprType::Name(_) => !self.builtin(&c.func).is_some_and(Builtin::pure),
                ExprType::Attribute(a) => !self.immutable_value(&a.value),
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

    /// Whether no part a comprehension evaluates per item can observe
    /// the order: the element(s), every condition, and every generator's
    /// iterable after the first (evaluated once per item of the ones
    /// before it), each in the scope its preceding generators bind.
    fn pure_comprehension(&self, elts: &[&ExprType], generators: &[Comprehension]) -> bool {
        let scopes = self.prefix_scopes(generators);
        let last = self.with(&scopes[generators.len()]);
        elts.iter().all(|e| last.pure(e))
            && generators.iter().enumerate().all(|(i, g)| {
                (i == 0 || self.with(&scopes[i]).pure(&g.iter))
                    && g.ifs.iter().all(|c| self.with(&scopes[i + 1]).pure(c))
            })
    }

    /// The type of the elements an iterable yields (a comprehension's
    /// element typed in its own scope).
    fn element_type(&self, e: &ExprType) -> Option<TypeInfo> {
        if let Some((elts, generators)) = comprehension(e)
            && !matches!(e, ExprType::DictComp(_))
        {
            let scopes = self.prefix_scopes(generators);
            return Some(self.with(&scopes[generators.len()]).type_of(elts[0]));
        }
        crate::ast::tree::type_ctx::iterable_element_type(&self.type_of(e))
    }

    /// Whether `call` is an order-free consumer of its iterable argument.
    fn order_free_call(&self, call: &crate::Call) -> bool {
        let Some(builtin) = self.builtin(&call.func).filter(|b| b.order_free()) else {
            return false;
        };
        let keyed = call
            .keywords
            .iter()
            .any(|k| k.arg.as_deref() == Some("key"));
        let elements = || call.args.first().and_then(|a| self.element_type(a));
        match builtin {
            // A NaN compares neither less nor greater, so with floats the
            // reduction (and the sort) keeps whichever came first.
            Builtin::Sorted | Builtin::Min | Builtin::Max => {
                !keyed && !elements().is_some_and(|t| may_hold_float(&t))
            }
            // Integer addition is exact in any order; float addition is not.
            Builtin::Sum => elements().is_some_and(|t| matches!(t, TypeInfo::Int | TypeInfo::Bool)),
            _ => true,
        }
    }
}

/// Whether values of type `t` can be (or hold) a float, whose NaN breaks
/// the total order sorting and min/max rely on.
fn may_hold_float(t: &TypeInfo) -> bool {
    match t {
        TypeInfo::Float => true,
        TypeInfo::Tuple(members) => members.iter().any(may_hold_float),
        TypeInfo::PyTuple(e) | TypeInfo::Option(e) | TypeInfo::Borrowed(e) => may_hold_float(e),
        _ => false,
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
         in, set operations, any/all, min/max/sorted without key= over \
         non-floats, sum of ints) — rython refuses to silently reorder it \
         (issue #444)",
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
    let scope = Scope {
        ctx,
        options,
        symbols,
    };
    let mut err: Option<String> = None;
    visit::walk_stmts(
        body,
        Descend::SkipDefs,
        &mut |s| match check_stmt(&scope, s) {
            Ok(()) => Flow::Continue,
            Err(e) => {
                err = Some(e);
                Flow::Stop
            }
        },
    );
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
            if a.targets
                .iter()
                .any(|t| matches!(t, ExprType::Tuple(_) | ExprType::List(_)))
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
        if let Some(what) = check_tree(scope, e, &mut allowed) {
            return Err(refusal(what, line));
        }
    }
    Ok(())
}

/// Check `e` and its subexpressions pre-order, entering each part of a
/// comprehension in the scope its preceding generators bind (so a
/// comprehension target that holds a set is known as one, and one that
/// shadows an outer set is not).
fn check_tree(
    scope: &Scope<'_>,
    e: &ExprType,
    allowed: &mut Vec<*const ExprType>,
) -> Option<&'static str> {
    if let Some(what) = check_expr(scope, e, allowed) {
        return Some(what);
    }
    if let Some((elts, generators)) = comprehension(e) {
        let scopes = scope.prefix_scopes(generators);
        for (i, g) in generators.iter().enumerate() {
            let at = scope.with(&scopes[i]);
            if let Some(what) = check_tree(&at, &g.iter, allowed) {
                return Some(what);
            }
            let inner = scope.with(&scopes[i + 1]);
            for c in &g.ifs {
                if let Some(what) = check_tree(&inner, c, allowed) {
                    return Some(what);
                }
            }
        }
        let last = scope.with(&scopes[generators.len()]);
        return elts.iter().find_map(|x| check_tree(&last, x, allowed));
    }
    let mut found = None;
    visit::each_subexpr(e, Descend::All, &mut |sub| {
        found = check_tree(scope, sub, allowed);
        found.is_none()
    });
    found
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
        && scope.builtin(&copy.func).is_some_and(Builtin::ordered_copy)
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
            if !set_arg {
                return None;
            }
            if let Some(builtin) = scope.builtin(&call.func) {
                let keyed = call
                    .keywords
                    .iter()
                    .any(|k| k.arg.as_deref() == Some("key"));
                return Some(builtin.refusal(keyed));
            }
            match call.func.as_ref() {
                ExprType::Name(n) if scope.is_builtin(&n.id) => {
                    Some("passing a set to an order-observing builtin")
                }
                // A set receiver's own methods are order-free (`s.pop()`
                // is checked at run time: `py_set_pop`).
                ExprType::Attribute(a)
                    if !scope.is_set(&a.value) && scope.builtin_receiver(&a.value) =>
                {
                    Some(match a.attr.as_str() {
                        "join" => "`str.join` over a set",
                        "extend" => "`list.extend` from a set",
                        "fromkeys" => "`dict.fromkeys` over a set",
                        _ => "passing a set to an order-observing method",
                    })
                }
                _ => None,
            }
        }
        ExprType::List(elts) if starred_set(elts) => Some("unpacking a set into a list (`[*s]`)"),
        ExprType::Tuple(t) if starred_set(&t.elts) => {
            Some("unpacking a set into a tuple (`(*s,)`)")
        }
        ExprType::YieldFrom(y) if scope.is_set(&y.value) => Some("`yield from` a set"),
        _ => {
            let (elts, generators) = comprehension(e)?;
            if !scope.iterates_set(generators) {
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
