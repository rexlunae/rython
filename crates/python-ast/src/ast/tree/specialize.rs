//! Conversion-time monomorphization of isinstance-dispatching functions.
//!
//! Python's `isinstance` dispatch idiom is inherently dynamic typing:
//!
//! ```python
//! def describe(x):
//!     if isinstance(x, int):
//!         return "int: " + str(x)
//!     if isinstance(x, Animal):
//!         return x.speak()
//!     return "other"
//! ```
//!
//! rython's value model has no runtime types to dispatch on, so instead
//! the isinstance checks become COMPILE-TIME flags: the converter emits
//! one specialized Rust function per input type (`describe_int`,
//! `describe_dog`, ...) plus a generic residual (`describe_any`), and
//! rewrites each call site to the variant matching the argument's static
//! type. Inside a variant the axis parameter is concretely typed, so
//! every isinstance folds to a constant through the inheritance tree
//! (see `ClassDef::class_extends`) and the dead arms are pruned before
//! they are ever rendered — the Rust equivalent of `if constexpr`, done
//! by the transpiler. Arms whose bodies would not even type-check for the
//! variant's type simply disappear, exactly like CPython never executing
//! them.
//!
//! Class arguments get PER-CONCRETE-CLASS variants (every module class
//! inside a tested subtree): Rust structs have no subtyping, so a `Cat`
//! cannot flow into an `Animal`-typed parameter — instead `describe_cat`
//! is emitted with `x: Cat`, where `isinstance(x, Animal)` folds true
//! through the tree and `x.speak()` keeps Cat's own override, exactly
//! like CPython. Python's first-true-test-wins branch order is preserved
//! by the folding (sound under single inheritance).
//!
//! Loud boundaries (never silently different):
//! - a call whose axis argument's type is statically unknown cannot be
//!   dispatched — conversion error naming the call;
//! - an isinstance test on an inferred-generic parameter in a shape this
//!   pass cannot specialize (a use outside a plain `if` test, a second
//!   tested parameter, defaults/varargs) — conversion error naming the
//!   construct.

use std::collections::HashMap;

use crate::ast::tree::statement::Statement;
use crate::ast::tree::StatementType;
use crate::{ExprType, SymbolTableNode, SymbolTableScopes};

/// One specialization target: a builtin type name or a user class.
#[derive(Clone, Debug, PartialEq)]
pub enum SpecTarget {
    /// `int`, `float`, `str`, `bool`, `bytes` — annotated verbatim.
    Builtin(String),
    /// A user class; dispatch walks the inheritance tree.
    Class(String),
}

impl SpecTarget {
    /// The type name this target annotates the axis with.
    pub fn suffix(&self) -> &str {
        match self {
            SpecTarget::Builtin(n) | SpecTarget::Class(n) => n,
        }
    }
}

/// The variant's Rust function name: lowercase suffix so class-named
/// variants (`describe_dog` for Dog) stay snake_case.
pub fn mangled_name(fn_name: &str, suffix: &str) -> String {
    format!("{}_{}", fn_name, suffix.to_lowercase())
}

/// A function the converter monomorphizes over one parameter.
#[derive(Clone, Debug)]
pub struct SpecializedFn {
    /// Index of the dispatched parameter in `args.args`.
    pub axis: usize,
    /// Its name (the isinstance-tested parameter).
    pub axis_name: String,
    /// The tested types, in first-test order (the folding truth table).
    pub targets: Vec<SpecTarget>,
    /// The CONCRETE classes that get a variant: every module class that
    /// is (or inherits from) a tested class. Rust structs have no
    /// subtyping, so a `Cat` argument cannot flow into an `Animal`-typed
    /// variant — instead `describe_Cat` is generated with `x: Cat`, where
    /// each isinstance folds through the inheritance tree and `x.speak()`
    /// keeps Cat's own override, exactly like CPython.
    pub class_variants: Vec<String>,
    /// The DYNAMIC router, when every morph's return type unifies: a
    /// closed-world argument enum (`DescribeArg`, one variant per morph
    /// plus `Other(PyValue)`) and a function under the ORIGINAL Python
    /// name that matches on it — runtime dispatch over the compile-time
    /// morphs, for boxed values and for Rust callers with runtime-varying
    /// data. None when the morphs' return types disagree or a name would
    /// collide; static call-site dispatch is unaffected either way.
    pub router: Option<RouterPlan>,
}

/// The emitted shape of a dynamic router (see `SpecializedFn::router`).
#[derive(Clone, Debug)]
pub struct RouterPlan {
    /// The argument-enum name (`DescribeArg`).
    pub enum_name: String,
    /// The unified Python return-type id of every morph ("str", "int",
    /// a class name, ...).
    pub return_id: String,
}

/// snake_case → PascalCase for the router enum and its variants.
pub fn to_pascal(name: &str) -> String {
    name.split('_')
        .filter(|seg| !seg.is_empty())
        .map(|seg| {
            let mut c = seg.chars();
            match c.next() {
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// The Rust type a Python type id lowers to in router positions.
pub fn py_type_tokens(id: &str) -> proc_macro2::TokenStream {
    use quote::quote;
    match id {
        "int" => quote!(i64),
        "float" => quote!(f64),
        "bool" => quote!(bool),
        "str" => quote!(String),
        "bytes" => quote!(Vec<u8>),
        class => {
            let ident = crate::safe_ident(class);
            quote!(#ident)
        }
    }
}

/// The registry of specializable module functions, built once per module
/// and carried in `PythonOptions` for call sites and the function
/// renderer.
pub type SpecRegistry = HashMap<String, SpecializedFn>;

/// The builtin names a specialization axis may test.
const SPEC_BUILTINS: &[&str] = &["int", "float", "str", "bool", "bytes"];

/// `isinstance(<name>, <Name target>)` as (arg, target), when the call is
/// exactly that shape.
fn isinstance_parts(expr: &ExprType) -> Option<(&str, &str)> {
    let ExprType::Call(c) = expr else { return None };
    let ExprType::Name(f) = c.func.as_ref() else {
        return None;
    };
    if f.id != "isinstance" || c.args.len() != 2 || !c.keywords.is_empty() {
        return None;
    }
    let ExprType::Name(arg) = &c.args[0] else {
        return None;
    };
    let ExprType::Name(target) = &c.args[1] else {
        return None;
    };
    Some((&arg.id, &target.id))
}

/// Every parameter name mentioned as the first argument of an isinstance
/// call ANYWHERE in the expression (not just plain if-tests).
fn isinstance_args_in_expr(expr: &ExprType, out: &mut Vec<String>) {
    if let Some((arg, _)) = isinstance_parts(expr) {
        out.push(arg.to_string());
        return;
    }
    match expr {
        ExprType::Call(c) => {
            isinstance_args_in_expr(&c.func, out);
            for a in &c.args {
                isinstance_args_in_expr(a, out);
            }
            for k in &c.keywords {
                isinstance_args_in_expr(&k.value, out);
            }
        }
        ExprType::BoolOp(b) => {
            for v in &b.values {
                isinstance_args_in_expr(v, out);
            }
        }
        ExprType::BinOp(b) => {
            isinstance_args_in_expr(&b.left, out);
            isinstance_args_in_expr(&b.right, out);
        }
        ExprType::UnaryOp(u) => isinstance_args_in_expr(&u.operand, out),
        ExprType::IfExp(i) => {
            isinstance_args_in_expr(&i.test, out);
            isinstance_args_in_expr(&i.body, out);
            isinstance_args_in_expr(&i.orelse, out);
        }
        ExprType::Compare(c) => {
            isinstance_args_in_expr(&c.left, out);
            for r in &c.comparators {
                isinstance_args_in_expr(r, out);
            }
        }
        ExprType::List(items) => {
            for e in items {
                isinstance_args_in_expr(e, out);
            }
        }
        ExprType::Tuple(t) => {
            for e in &t.elts {
                isinstance_args_in_expr(e, out);
            }
        }
        ExprType::Attribute(a) => isinstance_args_in_expr(&a.value, out),
        ExprType::Subscript(s) => isinstance_args_in_expr(&s.value, out),
        ExprType::NamedExpr(n) => {
            isinstance_args_in_expr(&n.left, out);
            isinstance_args_in_expr(&n.right, out);
        }
        _ => {}
    }
}

/// Decide whether `f` (a module-level function) is specializable, and on
/// which parameter. Returns None for shapes this pass does not cover —
/// those keep the ordinary lowering, where an isinstance on an inferred
/// parameter is a loud error.
pub fn detect_specializable(
    f: &crate::FunctionDef,
    symbols: &SymbolTableScopes,
    module_classes: &[String],
    options: &crate::PythonOptions,
) -> Option<SpecializedFn> {
    // Plain positional signatures only: dispatch rewrites call sites
    // positionally.
    if !f.args.posonlyargs.is_empty()
        || !f.args.defaults.is_empty()
        || f.args.vararg.is_some()
        || !f.args.kwonlyargs.is_empty()
        || f.args.kwarg.is_some()
        || !f.decorator_list.is_empty()
    {
        return None;
    }
    let unannotated: Vec<(usize, String)> = f
        .args
        .args
        .iter()
        .enumerate()
        .filter(|(_, p)| p.arg != "self" && p.annotation.is_none())
        .map(|(i, p)| (i, p.arg.clone()))
        .collect();
    if unannotated.is_empty() || f.args.args.iter().any(|p| p.arg == "self") {
        return None;
    }

    // Collect plain `if isinstance(p, T):` tests in statement order, and
    // every OTHER isinstance-of-a-parameter occurrence (which blocks
    // specialization: the residual could not eliminate it).
    let mut tests: Vec<(String, String)> = Vec::new(); // (param, target)
    let mut stray: Vec<String> = Vec::new();
    collect_isinstance_tests(&f.body, &unannotated, &mut tests, &mut stray);

    let axis = unannotated
        .iter()
        .find(|(_, name)| tests.iter().any(|(p, _)| p == name))?;
    // A second isinstance-tested unannotated parameter, or an isinstance
    // use outside a plain if-test: not specializable (the caller's
    // lowering reports the loud error).
    if tests.iter().any(|(p, _)| p != &axis.1)
        || stray.iter().any(|p| unannotated.iter().any(|(_, n)| n == p))
    {
        return None;
    }

    let mut targets: Vec<SpecTarget> = Vec::new();
    for (_, t) in tests.iter().filter(|(p, _)| p == &axis.1) {
        let target = if SPEC_BUILTINS.contains(&t.as_str()) {
            SpecTarget::Builtin(t.clone())
        } else if matches!(symbols.get(t), Some(SymbolTableNode::ClassDef(_))) {
            SpecTarget::Class(t.clone())
        } else {
            // An unresolvable target (an imported name, a tuple alias):
            // leave the shape to the ordinary lowering.
            return None;
        };
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    if targets.is_empty() {
        return None;
    }
    // Every concrete module class inside the tested subtrees gets its own
    // variant (see `class_variants` on SpecializedFn).
    let class_variants: Vec<String> = module_classes
        .iter()
        .filter(|d| {
            targets.iter().any(|t| match t {
                SpecTarget::Class(c) => {
                    crate::ast::tree::class_def::ClassDef::class_extends(d, c, symbols)
                }
                SpecTarget::Builtin(_) => false,
            })
        })
        .cloned()
        .collect();
    // The variant names must not collide with existing module items
    // (`describe` specializing to `describe_int` while the module also
    // defines a real `describe_int`): skip specialization — the ordinary
    // lowering reports the loud isinstance-on-generic error instead of a
    // confusing duplicate-definition failure.
    let mut mangled: Vec<String> = targets
        .iter()
        .filter(|t| matches!(t, SpecTarget::Builtin(_)))
        .map(|t| mangled_name(&f.name, t.suffix()))
        .collect();
    mangled.extend(class_variants.iter().map(|c| mangled_name(&f.name, c)));
    mangled.push(mangled_name(&f.name, "any"));
    let mut seen = std::collections::HashSet::new();
    if mangled.iter().any(|m| symbols.get(m).is_some() || !seen.insert(m.clone())) {
        return None;
    }
    let router = plan_router(f, &axis.1, &targets, &class_variants, symbols, options);
    Some(SpecializedFn {
        axis: axis.0,
        axis_name: axis.1.clone(),
        targets,
        class_variants,
        router,
    })
}

/// Walk statements collecting plain `if isinstance(p, T):` tests (in
/// order) and stray isinstance-of-parameter uses anywhere else.
fn collect_isinstance_tests(
    body: &[Statement],
    unannotated: &[(usize, String)],
    tests: &mut Vec<(String, String)>,
    stray: &mut Vec<String>,
) {
    let is_param = |name: &str| unannotated.iter().any(|(_, p)| p == name);
    for stmt in body {
        match &stmt.statement {
            StatementType::If(s) => {
                if let Some((arg, target)) = isinstance_parts(&s.test) {
                    if is_param(arg) {
                        tests.push((arg.to_string(), target.to_string()));
                    }
                } else {
                    isinstance_args_in_expr(&s.test, stray);
                }
                collect_isinstance_tests(&s.body, unannotated, tests, stray);
                collect_isinstance_tests(&s.orelse, unannotated, tests, stray);
            }
            other => visit_statement_exprs(other, &mut |e| {
                isinstance_args_in_expr(e, stray)
            }),
        }
    }
}

/// Visit the expressions directly held by a statement (shallow — nested
/// statement bodies are walked by the caller where needed; for stray
/// detection a full walk of non-If statements suffices via their nested
/// statements' own visits).
fn visit_statement_exprs(stmt: &StatementType, f: &mut impl FnMut(&ExprType)) {
    match stmt {
        StatementType::Expr(e) => f(&e.value),
        StatementType::Return(Some(r)) => f(&r.value),
        StatementType::Assign(a) => {
            f(&a.value);
            for t in &a.targets {
                f(t);
            }
        }
        StatementType::AugAssign(a) => {
            f(&a.target);
            f(&a.value);
        }
        StatementType::While(s) => {
            f(&s.test);
            for b in s.body.iter().chain(s.orelse.iter()) {
                visit_statement_exprs(&b.statement, f);
            }
        }
        StatementType::For(s) => {
            f(&s.iter);
            for b in s.body.iter().chain(s.orelse.iter()) {
                visit_statement_exprs(&b.statement, f);
            }
        }
        StatementType::With(s) => {
            for b in &s.body {
                visit_statement_exprs(&b.statement, f);
            }
        }
        StatementType::Try(t) => {
            for b in t
                .body
                .iter()
                .chain(t.orelse.iter())
                .chain(t.finalbody.iter())
            {
                visit_statement_exprs(&b.statement, f);
            }
            for h in &t.handlers {
                for b in &h.body {
                    visit_statement_exprs(&b.statement, f);
                }
            }
        }
        _ => {}
    }
}

/// The residual body: every plain `if isinstance(axis, T):` collapses to
/// its else-branch, because in the residual the axis is a type OUTSIDE
/// the tested set.
pub fn prune_axis_isinstance(body: &[Statement], axis: &str) -> Vec<Statement> {
    fold_axis_tests(body, axis, &|_| false)
}

/// Decide whether the dynamic router can exist: every morph (builtin
/// variants, per-class variants, and the residual) must derive the SAME
/// return type, and the enum/variant names must be free. Disagreeing
/// returns disable only the router — static dispatch is unaffected.
fn plan_router(
    f: &crate::FunctionDef,
    axis: &str,
    targets: &[SpecTarget],
    class_variants: &[String],
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Option<RouterPlan> {
    // Single-parameter functions only: a router for a multi-parameter
    // function would have to thread the other (possibly generic)
    // parameters through the enum signature.
    if f.args.args.len() != 1 {
        return None;
    }
    let morphs: Vec<SpecTarget> = targets
        .iter()
        .filter(|t| matches!(t, SpecTarget::Builtin(_)))
        .cloned()
        .chain(class_variants.iter().map(|c| SpecTarget::Class(c.clone())))
        .collect();
    let mut return_id: Option<String> = None;
    for morph in &morphs {
        let folded = fold_variant_body(&f.body, axis, morph, symbols);
        let id = derive_return_type_id(&folded, axis, Some(morph), options, symbols)?;
        match &return_id {
            None => return_id = Some(id),
            Some(prev) if *prev == id => {}
            Some(_) => return None,
        }
    }
    let residual = prune_axis_isinstance(&f.body, axis);
    let residual_id = derive_return_type_id(&residual, axis, None, options, symbols)?;
    let return_id = return_id?;
    if residual_id != return_id {
        return None;
    }
    let enum_name = format!("{}Arg", to_pascal(&f.name));
    if symbols.get(&enum_name).is_some() {
        return None;
    }
    // Variant idents must be distinct (two classes differing only in
    // case would collide after Pascal-casing the builtins).
    let mut idents = std::collections::HashSet::new();
    for morph in &morphs {
        let ident = match morph {
            SpecTarget::Builtin(b) => to_pascal(b),
            SpecTarget::Class(c) => c.clone(),
        };
        if !idents.insert(ident) {
            return None;
        }
    }
    if idents.contains("Other") {
        return None;
    }
    Some(RouterPlan { enum_name, return_id })
}

/// The compile-time truth of `isinstance(<axis>: <variant type>, T)` —
/// the same table the runtime fold in call.rs uses (bool ⊂ int; a class
/// answers through the inheritance tree).
fn variant_isinstance_taken(
    variant: &SpecTarget,
    target: &str,
    symbols: &SymbolTableScopes,
) -> bool {
    match variant {
        SpecTarget::Builtin(b) => b == target || (b == "bool" && target == "int"),
        SpecTarget::Class(c) => {
            crate::ast::tree::class_def::ClassDef::class_extends(c, target, symbols)
        }
    }
}

/// The variant body: every plain `if isinstance(axis, T):` is DECIDED for
/// the variant's concrete type — the live branch is spliced in place of
/// the if, the dead one dropped — so the variant reads like the
/// hand-written per-type function.
pub fn fold_variant_body(
    body: &[Statement],
    axis: &str,
    variant: &SpecTarget,
    symbols: &SymbolTableScopes,
) -> Vec<Statement> {
    fold_axis_tests(body, axis, &|target| {
        variant_isinstance_taken(variant, target, symbols)
    })
}

/// The shared transform: decide each plain `if isinstance(axis, T):` with
/// `taken`, splice the live branch, drop the dead one, recurse through
/// nested statement bodies, and truncate dead tails after an
/// unconditional return/raise (Python never runs them; emitting them
/// would only draw rustc's unreachable_code warning).
fn fold_axis_tests(
    body: &[Statement],
    axis: &str,
    taken: &dyn Fn(&str) -> bool,
) -> Vec<Statement> {
    let mut out = Vec::new();
    for stmt in body {
        let mut stmt = stmt.clone();
        match &mut stmt.statement {
            StatementType::If(s) => {
                if let Some((arg, target)) = isinstance_parts(&s.test) {
                    if arg == axis {
                        let branch = if taken(target) { &s.body } else { &s.orelse };
                        out.extend(fold_axis_tests(branch, axis, taken));
                        if ends_terminal(&out) {
                            break;
                        }
                        continue;
                    }
                }
                s.body = fold_axis_tests(&s.body, axis, taken);
                s.orelse = fold_axis_tests(&s.orelse, axis, taken);
            }
            StatementType::While(s) => {
                s.body = fold_axis_tests(&s.body, axis, taken);
                s.orelse = fold_axis_tests(&s.orelse, axis, taken);
            }
            StatementType::For(s) => {
                s.body = fold_axis_tests(&s.body, axis, taken);
                s.orelse = fold_axis_tests(&s.orelse, axis, taken);
            }
            StatementType::With(s) => {
                s.body = fold_axis_tests(&s.body, axis, taken);
            }
            StatementType::Try(t) => {
                t.body = fold_axis_tests(&t.body, axis, taken);
                t.orelse = fold_axis_tests(&t.orelse, axis, taken);
                t.finalbody = fold_axis_tests(&t.finalbody, axis, taken);
                for h in &mut t.handlers {
                    h.body = fold_axis_tests(&h.body, axis, taken);
                }
            }
            _ => {}
        }
        let terminal = matches!(
            stmt.statement,
            StatementType::Return(_) | StatementType::Raise(_)
        );
        out.push(stmt);
        if terminal {
            break;
        }
    }
    out
}

/// Whether the statement list already ends in an unconditional
/// return/raise (a spliced taken-branch that diverges).
fn ends_terminal(stmts: &[Statement]) -> bool {
    matches!(
        stmts.last().map(|s| &s.statement),
        Some(StatementType::Return(_)) | Some(StatementType::Raise(_))
    )
}

/// Derive a RETURN-TYPE annotation for a variant from its folded body:
/// the unified type of every returned expression, mapped back to a
/// Python annotation the ordinary annotated-function machinery already
/// understands (`-> str` owns literals, `-> Dog` names the struct).
/// None when returns are absent, mixed, or of a shape without a simple
/// annotation — the variant then keeps the default unit return, and a
/// real mismatch surfaces loudly in rustc.
pub fn derive_return_annotation(
    body: &[Statement],
    axis: &str,
    variant: &SpecTarget,
    options: &crate::PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<ExprType> {
    derive_return_type_id(body, axis, Some(variant), options, symbols)
        .map(|id| ExprType::Name(crate::ast::tree::name::Name { id }))
}

/// The unified Python type NAME of a morph body's returns ("str", "int",
/// a class name, ...), used both to annotate the emitted variant and to
/// decide whether the DYNAMIC ROUTER can exist (all morphs must agree).
/// `variant: None` types the residual, whose axis is unknown — returns
/// involving the axis then fail to unify, correctly disabling the router.
pub fn derive_return_type_id(
    body: &[Statement],
    axis: &str,
    variant: Option<&SpecTarget>,
    options: &crate::PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<String> {
    let mut opts = options.clone();
    let mut name_types = opts.name_types.as_ref().clone();
    if let Some(variant) = variant {
        let axis_ti = match variant {
            SpecTarget::Builtin(b) => match b.as_str() {
                "int" => crate::TypeInfo::Int,
                "float" => crate::TypeInfo::Float,
                "bool" => crate::TypeInfo::Bool,
                "str" => crate::TypeInfo::String,
                "bytes" => crate::TypeInfo::Bytes,
                _ => return None,
            },
            SpecTarget::Class(c) => crate::TypeInfo::Class(c.clone()),
        };
        name_types.insert(axis.to_string(), axis_ti);
    }
    let info = crate::analyze_function_types(body, Some(&opts), Some(symbols));
    for (k, v) in info.name_types {
        name_types.entry(k).or_insert(v);
    }
    opts.name_types = std::rc::Rc::new(name_types);

    let mut returns = Vec::new();
    collect_return_exprs(body, &mut returns);
    let mut unified: Option<crate::TypeInfo> = None;
    for r in &returns {
        let t = variant_expr_type(r, &opts, symbols);
        let t = match t {
            crate::TypeInfo::StrRef => crate::TypeInfo::String,
            other => other,
        };
        match &unified {
            None => unified = Some(t),
            Some(u) if *u == t => {}
            Some(_) => return None,
        }
    }
    let id = match unified? {
        crate::TypeInfo::Int => "int".to_string(),
        crate::TypeInfo::Float => "float".to_string(),
        crate::TypeInfo::Bool => "bool".to_string(),
        crate::TypeInfo::String => "str".to_string(),
        crate::TypeInfo::Bytes => "bytes".to_string(),
        crate::TypeInfo::Class(c) => c,
        _ => return None,
    };
    Some(id)
}

/// `infer_type` plus one rule it cannot state: `a + b` where EITHER side
/// is a string is a string (Python raises for str + non-str, and the
/// concat lowering already produces String) — this types the common
/// `x.name + ": " + x.speak()` return whose method-call side has no
/// statically-known type.
fn variant_expr_type(
    e: &ExprType,
    opts: &crate::PythonOptions,
    symbols: &SymbolTableScopes,
) -> crate::TypeInfo {
    if let ExprType::BinOp(b) = e
        && matches!(b.op, crate::ast::tree::BinOps::Add)
    {
        let stringy = |t: &crate::TypeInfo| {
            matches!(t, crate::TypeInfo::String | crate::TypeInfo::StrRef)
        };
        let l = variant_expr_type(&b.left, opts, symbols);
        let r = variant_expr_type(&b.right, opts, symbols);
        if stringy(&l) || stringy(&r) {
            return crate::TypeInfo::String;
        }
    }
    crate::infer_type(e, opts, symbols)
}

/// Every value-carrying `return` expression in the body (nested statement
/// bodies included; nested function definitions excluded).
fn collect_return_exprs(body: &[Statement], out: &mut Vec<ExprType>) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Return(Some(r)) => out.push(r.value.clone()),
            StatementType::If(s) => {
                collect_return_exprs(&s.body, out);
                collect_return_exprs(&s.orelse, out);
            }
            StatementType::While(s) => {
                collect_return_exprs(&s.body, out);
                collect_return_exprs(&s.orelse, out);
            }
            StatementType::For(s) => {
                collect_return_exprs(&s.body, out);
                collect_return_exprs(&s.orelse, out);
            }
            StatementType::With(s) => collect_return_exprs(&s.body, out),
            StatementType::Try(t) => {
                collect_return_exprs(&t.body, out);
                collect_return_exprs(&t.orelse, out);
                collect_return_exprs(&t.finalbody, out);
                for h in &t.handlers {
                    collect_return_exprs(&h.body, out);
                }
            }
            _ => {}
        }
    }
}

/// Underscore-prefix parameters a folded variant/residual body no longer
/// mentions, so the pruned definitions stay warning-clean. Conservative:
/// when the body contains a construct the mention-walk does not model,
/// nothing is renamed.
pub fn underscore_unused_params(f: &mut crate::FunctionDef) {
    let mut mentioned = std::collections::HashSet::new();
    if !collect_stmt_names(&f.body, &mut mentioned) {
        return;
    }
    for p in f.args.args.iter_mut() {
        if p.arg != "self" && !mentioned.contains(&p.arg) && !p.arg.starts_with('_') {
            p.arg = format!("_{}", p.arg);
        }
    }
}

/// Collect every Name mentioned in the statements; false when an
/// unmodeled construct is found (the caller must then assume everything
/// is mentioned).
fn collect_stmt_names(
    body: &[Statement],
    out: &mut std::collections::HashSet<String>,
) -> bool {
    for stmt in body {
        let ok = match &stmt.statement {
            StatementType::Expr(e) => collect_expr_names(&e.value, out),
            StatementType::Return(Some(r)) => collect_expr_names(&r.value, out),
            StatementType::Return(None)
            | StatementType::Pass
            | StatementType::Break
            | StatementType::Continue => true,
            StatementType::Assign(a) => {
                a.targets.iter().all(|t| collect_expr_names(t, out))
                    && collect_expr_names(&a.value, out)
            }
            StatementType::AugAssign(a) => {
                collect_expr_names(&a.target, out) && collect_expr_names(&a.value, out)
            }
            StatementType::AnnotatedName { .. } => true,
            StatementType::If(s) => {
                collect_expr_names(&s.test, out)
                    && collect_stmt_names(&s.body, out)
                    && collect_stmt_names(&s.orelse, out)
            }
            StatementType::While(s) => {
                collect_expr_names(&s.test, out)
                    && collect_stmt_names(&s.body, out)
                    && collect_stmt_names(&s.orelse, out)
            }
            StatementType::For(s) => {
                collect_expr_names(&s.target, out)
                    && collect_expr_names(&s.iter, out)
                    && collect_stmt_names(&s.body, out)
                    && collect_stmt_names(&s.orelse, out)
            }
            StatementType::Raise(r) => {
                r.exc.as_ref().map_or(true, |e| collect_expr_names(e, out))
                    && r.cause.as_ref().map_or(true, |e| collect_expr_names(e, out))
            }
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Collect every Name in the expression; false on an unmodeled variant.
fn collect_expr_names(
    e: &ExprType,
    out: &mut std::collections::HashSet<String>,
) -> bool {
    match e {
        ExprType::Name(n) => {
            out.insert(n.id.clone());
            true
        }
        ExprType::Constant(_) | ExprType::NoneType(_) => true,
        ExprType::Call(c) => {
            collect_expr_names(&c.func, out)
                && c.args.iter().all(|a| collect_expr_names(a, out))
                && c.keywords.iter().all(|k| collect_expr_names(&k.value, out))
        }
        ExprType::BoolOp(b) => b.values.iter().all(|v| collect_expr_names(v, out)),
        ExprType::BinOp(b) => {
            collect_expr_names(&b.left, out) && collect_expr_names(&b.right, out)
        }
        ExprType::UnaryOp(u) => collect_expr_names(&u.operand, out),
        ExprType::IfExp(i) => {
            collect_expr_names(&i.test, out)
                && collect_expr_names(&i.body, out)
                && collect_expr_names(&i.orelse, out)
        }
        ExprType::Compare(c) => {
            collect_expr_names(&c.left, out)
                && c.comparators.iter().all(|r| collect_expr_names(r, out))
        }
        ExprType::List(items) => items.iter().all(|e| collect_expr_names(e, out)),
        ExprType::Tuple(t) => t.elts.iter().all(|e| collect_expr_names(e, out)),
        ExprType::Attribute(a) => collect_expr_names(&a.value, out),
        ExprType::Subscript(s) => {
            collect_expr_names(&s.value, out)
                && match &s.kind {
                    crate::SubscriptKind::Index(i) => collect_expr_names(i, out),
                    crate::SubscriptKind::Slice { lower, upper, step } => [lower, upper, step]
                        .iter()
                        .all(|o| o.as_ref().map_or(true, |e| collect_expr_names(e, out))),
                }
        }
        ExprType::JoinedStr(j) => j.values.iter().all(|v| collect_expr_names(v, out)),
        ExprType::FormattedValue(f) => collect_expr_names(&f.value, out),
        _ => false,
    }
}

/// The dispatch decision for one call site: the mangling suffix of the
/// variant matching the argument's type, or None for the residual.
/// `arg_py_type` is the argument's Python type name ("int", "str", ...)
/// or user class name. A class argument dispatches to ITS OWN variant
/// (Rust structs have no subtyping), which exists exactly when the class
/// sits inside a tested subtree.
pub fn dispatch_suffix<'a>(
    spec: &'a SpecializedFn,
    arg_py_type: &str,
    arg_is_class: bool,
) -> Option<&'a str> {
    if arg_is_class {
        return spec
            .class_variants
            .iter()
            .find(|c| c.as_str() == arg_py_type)
            .map(String::as_str);
    }
    spec.targets
        .iter()
        .find(|t| matches!(t, SpecTarget::Builtin(b) if b == arg_py_type))
        .map(|t| t.suffix())
}
