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

/// One isinstance-tested parameter of a specialized function.
#[derive(Clone, Debug)]
pub struct SpecAxis {
    /// Index of the dispatched parameter in `args.args`.
    pub index: usize,
    /// Its name (the isinstance-tested parameter).
    pub name: String,
    /// The tested types, in first-test order (the folding truth table).
    pub targets: Vec<SpecTarget>,
    /// The CONCRETE classes that get a variant: every module class that
    /// is (or inherits from) a tested class. Rust structs have no
    /// subtyping, so a `Cat` argument cannot flow into an `Animal`-typed
    /// variant — instead `describe_Cat` is generated with `x: Cat`, where
    /// each isinstance folds through the inheritance tree and `x.speak()`
    /// keeps Cat's own override, exactly like CPython.
    pub class_variants: Vec<String>,
}

impl SpecAxis {
    /// This axis's variant list, in dispatch order: tested builtins,
    /// then the per-concrete-class variants.
    pub fn variants(&self) -> Vec<SpecTarget> {
        self.targets
            .iter()
            .filter(|t| matches!(t, SpecTarget::Builtin(_)))
            .cloned()
            .chain(
                self.class_variants
                    .iter()
                    .map(|c| SpecTarget::Class(c.clone())),
            )
            .collect()
    }
}

/// A function the converter monomorphizes over its isinstance-tested
/// parameters. With one axis the morphs are that axis's variants plus
/// the `_any` residual; with several axes the morphs are the CARTESIAN
/// PRODUCT over each axis of (its variants + Any), named
/// `f_{s1}_{s2}_...` in axis order (`f_str_int`, `f_str_any`,
/// `f_any_any`, ...), so a call site dispatches each argument
/// independently through its own axis.
#[derive(Clone, Debug)]
pub struct SpecializedFn {
    /// The isinstance-tested parameters, in parameter order.
    pub axes: Vec<SpecAxis>,
    /// The DYNAMIC router: per-axis argument enums (`DescribeArg`, or
    /// `DescribeArg1`/`DescribeArg2`/... with several axes, each with
    /// one variant per axis variant plus `Other(PyValue)`) and a
    /// function under the ORIGINAL Python name that tuple-matches them
    /// — runtime dispatch over the compile-time morphs, for boxed
    /// values and for Rust callers with runtime-varying data. None when
    /// a morph's return type cannot be derived, a non-axis parameter
    /// lacks a concrete annotation, or a name would collide; static
    /// call-site dispatch is unaffected either way.
    pub router: Option<RouterPlan>,
}

impl SpecializedFn {
    /// Every morph as one assignment per axis (None = the axis is a
    /// type OUTSIDE its tested set), in dispatch order; the all-None
    /// entry is the full residual. The cartesian product over axes.
    pub fn morph_assignments(&self) -> Vec<Vec<Option<SpecTarget>>> {
        let mut combos: Vec<Vec<Option<SpecTarget>>> = vec![Vec::new()];
        for axis in &self.axes {
            let mut next = Vec::new();
            for combo in &combos {
                for v in axis.variants() {
                    let mut c = combo.clone();
                    c.push(Some(v));
                    next.push(c);
                }
                let mut c = combo.clone();
                c.push(None);
                next.push(c);
            }
            combos = next;
        }
        combos
    }

    /// The mangling suffix of one morph assignment (`str_int`,
    /// `any_int`, `any` for the single-axis residual, ...).
    pub fn assignment_suffix(assignment: &[Option<SpecTarget>]) -> String {
        assignment
            .iter()
            .map(|a| match a {
                Some(t) => t.suffix().to_lowercase(),
                None => "any".to_string(),
            })
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// The emitted shape of a dynamic router (see `SpecializedFn::router`).
#[derive(Clone, Debug)]
pub struct RouterPlan {
    /// The per-axis argument-enum names, parallel to `axes`
    /// (`DescribeArg`, or `DescribeArg1`/`DescribeArg2`/... with
    /// several axes).
    pub enum_names: Vec<String>,
    /// The non-axis parameters, in signature order: (index in
    /// `args.args`, name, Python type id from the annotation). They pass
    /// through the router unchanged — no enum needed for an untested
    /// parameter.
    pub extra_params: Vec<(usize, String, String)>,
    /// What the router returns.
    pub ret: RouterReturn,
}

/// The router's return shape.
#[derive(Clone, Debug)]
pub enum RouterReturn {
    /// Every morph (and the residual) derived the SAME Python return
    /// type — the router returns it directly.
    Unified(String),
    /// The morphs' return types differ: the router returns an OUTPUT
    /// enum (`DescribeOut`) with one variant per distinct return type
    /// and `From<T>` per member, so a runtime-dispatched result lands
    /// as a value the caller can match on (and, when every member is
    /// boxable, convert to `PyValue` — Python's `str | int` union).
    Enum {
        /// The output-enum name (`DescribeOut`).
        name: String,
        /// The distinct Python return-type ids, in morph order.
        members: Vec<String>,
    },
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

    // An isinstance use outside a plain if-test on ANY unannotated
    // parameter: not specializable (the caller's lowering reports the
    // loud error).
    if stray.iter().any(|p| unannotated.iter().any(|(_, n)| n == p)) {
        return None;
    }
    // Every isinstance-tested unannotated parameter becomes an AXIS, in
    // parameter order; the morphs are the cartesian product over axes.
    let mut axes: Vec<SpecAxis> = Vec::new();
    for (index, name) in &unannotated {
        if !tests.iter().any(|(p, _)| p == name) {
            continue;
        }
        let mut targets: Vec<SpecTarget> = Vec::new();
        for (_, t) in tests.iter().filter(|(p, _)| p == name) {
            let target = if SPEC_BUILTINS.contains(&t.as_str()) {
                SpecTarget::Builtin(t.clone())
            } else if matches!(symbols.get(t), Some(SymbolTableNode::ClassDef(_))) {
                SpecTarget::Class(t.clone())
            } else {
                // An unresolvable target (an imported name, a tuple
                // alias): leave the shape to the ordinary lowering.
                return None;
            };
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        // bool ⊂ int (CPython: `isinstance(True, int)` is True): an int
        // test also captures bool arguments, so an int-tested axis gets
        // a bool morph of its own even when bool is never tested by
        // name. Its folded body takes the int arms while the parameter
        // STAYS a Rust bool, so `str(x)` renders True/False exactly like
        // CPython — and a boxed bool routes through the router to this
        // morph, not a coerced int.
        let tests_int = targets
            .iter()
            .any(|t| matches!(t, SpecTarget::Builtin(b) if b == "int"));
        let tests_bool = targets
            .iter()
            .any(|t| matches!(t, SpecTarget::Builtin(b) if b == "bool"));
        if tests_int && !tests_bool {
            targets.push(SpecTarget::Builtin("bool".to_string()));
        }
        // Every concrete module class inside the tested subtrees gets
        // its own variant (see `class_variants` on SpecAxis).
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
        axes.push(SpecAxis {
            index: *index,
            name: name.clone(),
            targets,
            class_variants,
        });
    }
    if axes.is_empty() {
        return None;
    }
    let spec = SpecializedFn { axes, router: None };
    let assignments = spec.morph_assignments();
    // The cartesian product multiplies: cap the morph count so a
    // many-axes many-classes function cannot explode the generated
    // module. Over the cap the whole shape falls back to the ordinary
    // lowering (the documented warn+false class-as-value divergence).
    const MORPH_CAP: usize = 32;
    if assignments.len() > MORPH_CAP {
        options.definition_warnings.borrow_mut().push(format!(
            "`{}` isinstance-dispatches over {} parameter(s) with {} morph \
             combinations (cap {}); not specialized",
            f.name,
            spec.axes.len(),
            assignments.len(),
            MORPH_CAP
        ));
        return None;
    }
    // The variant names must not collide with existing module items
    // (`describe` specializing to `describe_int` while the module also
    // defines a real `describe_int`): skip specialization — the ordinary
    // lowering reports the loud isinstance-on-generic error instead of a
    // confusing duplicate-definition failure.
    let mut seen = std::collections::HashSet::new();
    for a in &assignments {
        let m = mangled_name(&f.name, &SpecializedFn::assignment_suffix(a));
        if symbols.get(&m).is_some() || !seen.insert(m) {
            return None;
        }
    }
    let router = plan_router(f, &spec, symbols, options);
    Some(SpecializedFn { router, ..spec })
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

/// The TypeInfo a Python type id seeds return derivation with.
fn py_id_typeinfo(id: &str, symbols: &SymbolTableScopes) -> Option<crate::TypeInfo> {
    match id {
        "int" => Some(crate::TypeInfo::Int),
        "float" => Some(crate::TypeInfo::Float),
        "bool" => Some(crate::TypeInfo::Bool),
        "str" => Some(crate::TypeInfo::String),
        "bytes" => Some(crate::TypeInfo::Bytes),
        class if matches!(symbols.get(class), Some(SymbolTableNode::ClassDef(_))) => {
            Some(crate::TypeInfo::Class(class.to_string()))
        }
        _ => None,
    }
}

/// Whether a Python return-type id can live inside the boxed PyValue.
pub fn py_id_boxable(id: &str) -> bool {
    matches!(id, "int" | "float" | "bool" | "str" | "bytes")
}

/// The TypeInfo a morph assignment gives one axis.
pub fn target_typeinfo(t: &SpecTarget) -> Option<crate::TypeInfo> {
    match t {
        SpecTarget::Builtin(b) => match b.as_str() {
            "int" => Some(crate::TypeInfo::Int),
            "float" => Some(crate::TypeInfo::Float),
            "bool" => Some(crate::TypeInfo::Bool),
            "str" => Some(crate::TypeInfo::String),
            "bytes" => Some(crate::TypeInfo::Bytes),
            _ => None,
        },
        SpecTarget::Class(c) => Some(crate::TypeInfo::Class(c.clone())),
    }
}

/// Decide whether the dynamic router can exist: every morph must derive
/// a return type, every NON-axis parameter must carry a concrete
/// annotation (a builtin scalar or a module class — it passes through
/// the router unchanged), and the enum/variant names must be free.
/// Morphs whose return types DISAGREE still get a router: it returns an
/// output enum (`RouterReturn::Enum`) instead of the unified type. An
/// underivable shape disables only the router — static dispatch is
/// unaffected.
fn plan_router(
    f: &crate::FunctionDef,
    spec: &SpecializedFn,
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Option<RouterPlan> {
    // Every non-axis parameter passes through the router positionally,
    // so it needs ONE concrete type all morphs share: its annotation.
    // (An unannotated extra is inferred per-morph, possibly generically —
    // no single router signature covers that.)
    let mut extra_params: Vec<(usize, String, String)> = Vec::new();
    let mut extra_seeds: Vec<(String, crate::TypeInfo)> = Vec::new();
    for (i, p) in f.args.args.iter().enumerate() {
        if spec.axes.iter().any(|a| a.name == p.arg) {
            continue;
        }
        let Some(ExprType::Name(ann)) = p.annotation.as_deref() else {
            return None;
        };
        let ti = py_id_typeinfo(&ann.id, symbols)?;
        extra_seeds.push((p.arg.clone(), ti));
        extra_params.push((i, p.arg.clone(), ann.id.clone()));
    }
    // The distinct morph return ids, in morph order (the all-any
    // residual comes last in the cartesian product).
    let mut members: Vec<String> = Vec::new();
    for assignment in spec.morph_assignments() {
        let folded = fold_morph_body(&f.body, spec, &assignment, symbols);
        let mut seeds = extra_seeds.clone();
        for (axis, a) in spec.axes.iter().zip(&assignment) {
            if let Some(t) = a {
                seeds.push((axis.name.clone(), target_typeinfo(t)?));
            }
        }
        let id = derive_return_type_id(&folded, &seeds, options, symbols)?;
        if !members.contains(&id) {
            members.push(id);
        }
    }
    if members.is_empty() {
        return None;
    }
    // Per-axis argument enums: `DescribeArg` for a single axis,
    // `DescribeArg1`/`DescribeArg2`/... (numbered by parameter order)
    // with several.
    let enum_names: Vec<String> = if spec.axes.len() == 1 {
        vec![format!("{}Arg", to_pascal(&f.name))]
    } else {
        (1..=spec.axes.len())
            .map(|i| format!("{}Arg{}", to_pascal(&f.name), i))
            .collect()
    };
    let mut names_seen = std::collections::HashSet::new();
    for n in &enum_names {
        if symbols.get(n).is_some() || !names_seen.insert(n.clone()) {
            return None;
        }
    }
    // Variant idents must be distinct per axis (two classes differing
    // only in case would collide after Pascal-casing the builtins).
    for axis in &spec.axes {
        let mut idents = std::collections::HashSet::new();
        for v in axis.variants() {
            let ident = match &v {
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
    }
    let ret = if members.len() == 1 {
        RouterReturn::Unified(members.remove(0))
    } else {
        // Diverging morph returns: an output enum, one variant per
        // distinct return type, `From<T>` per member.
        let out_name = format!("{}Out", to_pascal(&f.name));
        if symbols.get(&out_name).is_some() || enum_names.contains(&out_name) {
            return None;
        }
        let mut out_idents = std::collections::HashSet::new();
        for m in &members {
            let ident = if py_id_boxable(m) { to_pascal(m) } else { m.clone() };
            if !out_idents.insert(ident) {
                return None;
            }
        }
        RouterReturn::Enum { name: out_name, members }
    };
    Some(RouterPlan { enum_names, extra_params, ret })
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

/// One MORPH's body: fold each axis in turn — an assigned axis decides
/// its tests through the truth table, an unassigned (Any) axis collapses
/// every test to its else-branch (the residual behavior for that axis).
pub fn fold_morph_body(
    body: &[Statement],
    spec: &SpecializedFn,
    assignment: &[Option<SpecTarget>],
    symbols: &SymbolTableScopes,
) -> Vec<Statement> {
    let mut folded = body.to_vec();
    for (axis, a) in spec.axes.iter().zip(assignment) {
        folded = match a {
            Some(v) => fold_variant_body(&folded, &axis.name, v, symbols),
            None => prune_axis_isinstance(&folded, &axis.name),
        };
    }
    folded
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
    seed_types: &[(String, crate::TypeInfo)],
    options: &crate::PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<ExprType> {
    derive_return_type_id(body, seed_types, options, symbols)
        .map(|id| ExprType::Name(crate::ast::tree::name::Name { id }))
}

/// The unified Python type NAME of a morph body's returns ("str", "int",
/// a class name, ...), used both to annotate the emitted variant and to
/// plan the DYNAMIC ROUTER's return shape. `seed_types` types the
/// morph's assigned axes and the annotated extra parameters; an
/// unassigned (Any) axis has no seed, so returns involving it fail to
/// derive — correctly disabling the router for that shape.
pub fn derive_return_type_id(
    body: &[Statement],
    seed_types: &[(String, crate::TypeInfo)],
    options: &crate::PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<String> {
    let mut opts = options.clone();
    let mut name_types = opts.name_types.as_ref().clone();
    for (name, ti) in seed_types {
        name_types.insert(name.clone(), ti.clone());
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
    // `x.decode(...)` is a str whatever the receiver's type: only bytes
    // has decode in Python 3, and the PyDecode lowering returns String.
    // This types the RESIDUAL morph of a str-tested dispatcher (`return
    // path.decode(fs_enc, 'replace')` — botocore configloader's
    // `_unicode_path`, issue #161), whose receiver has no static type,
    // so the router's return derives and the boxed fallback can exist.
    if let ExprType::Call(c) = e
        && let ExprType::Attribute(a) = c.func.as_ref()
        && a.attr == "decode"
    {
        return crate::TypeInfo::String;
    }
    crate::infer_type(e, opts, symbols)
}

/// Every value-carrying `return` expression in the body (nested statement
/// bodies included; nested function definitions excluded).
pub(crate) fn collect_return_exprs(body: &[Statement], out: &mut Vec<ExprType>) {
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

/// The dispatch decision for ONE AXIS of a call site: the suffix of the
/// variant matching the argument's type, or None for that axis's
/// residual ("any"). `arg_py_type` is the argument's Python type name
/// ("int", "str", ...) or user class name. A class argument dispatches
/// to ITS OWN variant (Rust structs have no subtyping), which exists
/// exactly when the class sits inside a tested subtree.
pub fn axis_dispatch_suffix<'a>(
    axis: &'a SpecAxis,
    arg_py_type: &str,
    arg_is_class: bool,
) -> Option<&'a str> {
    if arg_is_class {
        return axis
            .class_variants
            .iter()
            .find(|c| c.as_str() == arg_py_type)
            .map(String::as_str);
    }
    axis.targets
        .iter()
        .find(|t| matches!(t, SpecTarget::Builtin(b) if b == arg_py_type))
        .map(|t| t.suffix())
}
