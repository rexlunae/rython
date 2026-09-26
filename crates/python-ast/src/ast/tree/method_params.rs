//! Call-site inference of unannotated METHOD parameters (issue #335).
//!
//! An unannotated method parameter lowers to the boxed `PyValue`, which has
//! no fields: `def apply(self, order): order.discount = ...` cannot read or
//! store what every caller hands it. Python needs no annotation because the
//! object is whatever the caller passed. This pass asks the callers.
//!
//! The unit is a METHOD SLOT: a method name in a class family, keyed by the
//! topmost class of the chain that defines it. A base definition and its
//! overrides share one signature (the trait machinery requires it), so they
//! share one answer. The evidence is every call in the crate that resolves
//! to the slot — `self.m(x)`, `obj.m(x)` with `obj`'s class known,
//! `super().m(x)` — typed by the same analysis the code generator runs.
//! Types flow through chains of calls to a fixpoint: an argument that is
//! itself an unannotated parameter carries whatever that parameter infers.
//!
//! A parameter takes a class `X` only when EVERY resolved call passes an
//! `X`. Anything else leaves it boxed, exactly as before: no resolved call,
//! a disagreement, an argument whose type is unknown, a spread (`*args`,
//! `**kw`) or an unbound call (`Base.m(obj, x)`) into the slot, the method
//! referenced as a value, a default value, a decorator, a dunder (Python
//! calls those implicitly), a parameter the body rebinds, or a class the
//! method's own module cannot name.
//!
//! The answer is applied as a SYNTHESIZED ANNOTATION (`order: Order`): the
//! signature, the body's typing, call-site coercion, and the shared-class
//! decision (shared.rs — a mutated class held by a parameter shares the
//! caller's object) then take the annotated path, unchanged. It is computed
//! once per conversion over every module of the crate and applied to each
//! module's AST before anything reads it (module.rs).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use crate::ast::tree::visit::{self, Descend, Flow};
use crate::{
    ClassDef, CodeGen, CodeGenContext, ExprType, FunctionDef, PythonOptions, Statement,
    StatementType, SymbolTableScopes, TypeInfo,
};

/// `(the class whose body defines the method, method, parameter)` → the
/// class every call site passes.
pub type InferredParams = HashMap<(String, String, String), String>;

/// The once-per-conversion computation, shared by every module's options.
#[derive(Clone, Debug, Default)]
pub enum InferredParamsState {
    #[default]
    Uncomputed,
    /// Being computed: a re-entrant query answers with nothing.
    Computing,
    Computed(Rc<InferredParams>),
}

/// The crate's inferred method parameters, computed on first use.
/// `this_module` / `this_symbols` stand in for the whole crate when the
/// conversion has one module (an empty `module_defs`).
pub fn inferred_params(
    options: &PythonOptions,
    this_module: &crate::Module,
    this_symbols: &SymbolTableScopes,
) -> Rc<InferredParams> {
    match &*options.inferred_method_params.borrow() {
        InferredParamsState::Computed(map) => return map.clone(),
        InferredParamsState::Computing => return Rc::new(HashMap::new()),
        InferredParamsState::Uncomputed => {}
    }
    *options.inferred_method_params.borrow_mut() = InferredParamsState::Computing;
    let map = Rc::new(compute(options, this_module, this_symbols));
    *options.inferred_method_params.borrow_mut() = InferredParamsState::Computed(map.clone());
    // A cross-module class cache filled before now holds un-annotated
    // classes: the next lookup rebuilds it from the annotated modules.
    *options.cross_module_classes.borrow_mut() = crate::CrossModuleClasses::Uncomputed;
    *options.cross_module_mut_self.borrow_mut() = crate::CrossModuleMutSelf::Uncomputed;
    map
}

/// The computed map, when the computation has run (the cross-module class
/// cache applies it to each module it loads).
pub fn computed(options: &PythonOptions) -> Option<Rc<InferredParams>> {
    match &*options.inferred_method_params.borrow() {
        InferredParamsState::Computed(map) => Some(map.clone()),
        _ => None,
    }
}

/// Write the inferred annotations into every class's methods in `body`.
/// Returns whether anything changed.
pub fn annotate(body: &mut [Statement], map: &InferredParams) -> bool {
    if map.is_empty() {
        return false;
    }
    let mut changed = false;
    visit::walk_stmts_mut(body, &mut |s| {
        if let StatementType::ClassDef(c) = &mut s.statement {
            let class = c.name.clone();
            for m in c.body.iter_mut() {
                let (StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f)) =
                    &mut m.statement
                else {
                    continue;
                };
                let method = f.name.clone();
                let args = &mut f.args;
                for p in args
                    .posonlyargs
                    .iter_mut()
                    .chain(args.args.iter_mut())
                    .chain(args.kwonlyargs.iter_mut())
                {
                    if p.annotation.is_none()
                        && let Some(x) = map.get(&(class.clone(), method.clone(), p.arg.clone()))
                    {
                        p.annotation = Some(Box::new(ExprType::Name(crate::ast::tree::name::Name {
                            id: x.clone(),
                        })));
                        changed = true;
                    }
                }
            }
        }
        Flow::Continue
    });
    changed
}

/// One module of the crate, as the pass reads it.
struct Unit {
    module: crate::Module,
    symbols: SymbolTableScopes,
    options: PythonOptions,
}

/// A method slot: the topmost definer and the method name.
type SlotKey = (String, String);

/// The parameters of one definition in a slot, receiver excluded, in
/// call order (positional, then keyword-only), each with whether it may be
/// inferred.
struct SlotDef {
    class: String,
    params: Vec<(String, bool)>,
}

/// What one call site says about one parameter position.
#[derive(Clone, Debug, PartialEq)]
enum Evidence {
    /// A value of this class.
    Class(String),
    /// A parameter of the calling method that is itself still being
    /// inferred: agrees with anything, until the fixpoint settles it.
    Pending,
    /// Anything else: an unknown, boxed, primitive or optional value.
    Other,
}

#[derive(Default)]
struct Collected {
    evidence: HashMap<(SlotKey, usize), Vec<Evidence>>,
    poisoned: HashSet<SlotKey>,
}

fn compute(
    options: &PythonOptions,
    this_module: &crate::Module,
    this_symbols: &SymbolTableScopes,
) -> InferredParams {
    // A private view: the analysis below fills the cross-module caches with
    // the modules as they are now, which the real conversion must not see.
    let mut base = options.clone();
    base.cross_module_classes =
        Rc::new(std::cell::RefCell::new(crate::CrossModuleClasses::Uncomputed));
    base.cross_module_mut_self =
        Rc::new(std::cell::RefCell::new(crate::CrossModuleMutSelf::Uncomputed));
    let units: Vec<Unit> = if base.module_defs.is_empty() {
        vec![Unit {
            module: this_module.clone(),
            symbols: this_symbols.clone(),
            options: base.clone(),
        }]
    } else {
        let mut units = Vec::new();
        let mut paths: Vec<&Vec<String>> = base.module_defs.keys().collect();
        paths.sort();
        for path in paths {
            let module: &crate::Module = &base.module_defs[path];
            let mut module_opts = base.clone();
            let is_package = base
                .module_defs
                .keys()
                .any(|k| k.len() > path.len() && k[..path.len()] == path[..]);
            module_opts.module_path = if is_package {
                path.clone()
            } else {
                path[..path.len().saturating_sub(1)].to_vec()
            };
            module_opts.this_module_path = path.clone();
            let symbols = module.clone().find_symbols(SymbolTableScopes::new());
            units.push(Unit {
                module: module.clone(),
                symbols,
                options: module_opts,
            });
        }
        units
    };
    // The class hierarchy the typing consults (receiver resolution, sum
    // types), from the crate's classes; each module's conversion installs
    // its own view again afterwards.
    let mut roots_opts = base.clone();
    if !base.module_defs.is_empty() {
        roots_opts.this_module_path = vec![String::new()];
    }
    let (roots_body, roots_items) = if base.module_defs.is_empty() {
        let items = crate::ast::tree::module::emitted_class_defs(this_module, &base);
        (this_module.raw.body.clone(), items)
    } else {
        (Vec::new(), Vec::new())
    };
    let roots = crate::ast::tree::hierarchy::compute_roots(&roots_body, &roots_items, &roots_opts);
    crate::ast::tree::hierarchy::install_roots(&roots);
    let roots = Rc::new(roots);
    let units: Vec<Unit> = units
        .into_iter()
        .map(|mut u| {
            u.options.hierarchy_roots = roots.clone();
            u
        })
        .collect();

    // The crate's classes by bare name; a name two modules define is
    // ambiguous and takes no part (the hierarchy index applies the same
    // rule).
    let mut defined_in: HashMap<String, usize> = HashMap::new();
    let mut unit_classes: Vec<Vec<ClassDef>> = Vec::new();
    for u in &units {
        let classes = crate::ast::tree::module::emitted_class_defs(&u.module, &u.options);
        for c in &classes {
            *defined_in.entry(c.name.clone()).or_insert(0) += 1;
        }
        unit_classes.push(classes);
    }
    let unambiguous = |name: &str| defined_in.get(name).copied() == Some(1);

    // The candidate slots.
    let mut slots: BTreeMap<SlotKey, Vec<SlotDef>> = BTreeMap::new();
    let mut slot_units: HashMap<String, usize> = HashMap::new();
    for (ui, (u, classes)) in units.iter().zip(unit_classes.iter()).enumerate() {
        for c in classes {
            if !unambiguous(&c.name) {
                continue;
            }
            slot_units.insert(c.name.clone(), ui);
            for m in c.methods() {
                let Some(params) = instance_params(m) else {
                    continue;
                };
                let Some(root) = slot_root(c, &m.name, &u.symbols, &u.options) else {
                    continue;
                };
                slots.entry((root, m.name.clone())).or_default().push(SlotDef {
                    class: c.name.clone(),
                    params,
                });
            }
        }
    }
    // A position is inferable when EVERY definition of the slot has an
    // inferable parameter there (the override check requires one
    // signature for the family).
    let inferable: HashMap<SlotKey, Vec<usize>> = slots
        .iter()
        .map(|(key, defs)| {
            let width = defs.iter().map(|d| d.params.len()).min().unwrap_or(0);
            let positions = (0..width)
                .filter(|&i| defs.iter().all(|d| d.params[i].1))
                .collect();
            (key.clone(), positions)
        })
        .filter(|(_, positions): &(SlotKey, Vec<usize>)| !positions.is_empty())
        .collect();
    if inferable.is_empty() {
        return HashMap::new();
    }

    // The fixpoint over slot positions: grow while every call site agrees
    // (a pending caller parameter agrees with anything), then verify that
    // every call site passes exactly the class, removing until stable.
    let mut solved: HashMap<(SlotKey, usize), String> = HashMap::new();
    for _ in 0..8 {
        let collected = collect(&units, &inferable, &slots, &solved, &unambiguous);
        let next = decide(&collected, &inferable, true);
        if next == solved {
            break;
        }
        solved = next;
    }
    loop {
        let collected = collect(&units, &inferable, &slots, &solved, &unambiguous);
        let verified: HashMap<(SlotKey, usize), String> = decide(&collected, &inferable, false)
            .into_iter()
            .filter(|(k, v)| solved.get(k) == Some(v))
            .collect();
        if verified == solved {
            break;
        }
        solved = verified;
    }

    // Each definition's parameter at a solved position, when the class is
    // nameable in that definition's module.
    let mut out = InferredParams::new();
    for ((key, pos), class) in &solved {
        let defs = &slots[key];
        let nameable = defs.iter().all(|d| {
            slot_units.get(&d.class).is_some_and(|&ui| {
                let u = &units[ui];
                let ann = ExprType::Name(crate::ast::tree::name::Name { id: class.clone() });
                matches!(
                    crate::resolve_alias_typeinfo(&ann, &u.symbols, &u.options),
                    Some(TypeInfo::Class(c)) if c == *class
                )
            })
        });
        if !nameable {
            continue;
        }
        for d in defs {
            out.insert((d.class.clone(), key.1.clone(), d.params[*pos].0.clone()), class.clone());
        }
    }
    out
}

/// The slot positions whose evidence agrees on one class. `optimistic`
/// lets a pending caller parameter agree; otherwise every call site must
/// pass the class itself.
fn decide(
    collected: &Collected,
    inferable: &HashMap<SlotKey, Vec<usize>>,
    optimistic: bool,
) -> HashMap<(SlotKey, usize), String> {
    let mut out = HashMap::new();
    for (key, positions) in inferable {
        if collected.poisoned.contains(key) {
            continue;
        }
        for &pos in positions {
            let Some(evidence) = collected.evidence.get(&(key.clone(), pos)) else {
                continue;
            };
            let mut class: Option<&str> = None;
            let mut agrees = true;
            for e in evidence {
                match e {
                    Evidence::Class(c) => match class {
                        None => class = Some(c),
                        Some(prev) if prev == c => {}
                        Some(_) => agrees = false,
                    },
                    Evidence::Pending if optimistic => {}
                    Evidence::Pending | Evidence::Other => agrees = false,
                }
            }
            if let (true, Some(c)) = (agrees, class) {
                out.insert((key.clone(), pos), c.to_string());
            }
        }
    }
    out
}

/// A method's parameters after the receiver, with whether each may be
/// inferred. None for what is not a plain instance method: a static or
/// class method, a decorated method (a property, a setter, a cache — Python
/// calls those through the decorator), or a dunder (called implicitly).
fn instance_params(m: &FunctionDef) -> Option<Vec<(String, bool)>> {
    if !m.decorator_list.is_empty() || (m.name.starts_with("__") && m.name.ends_with("__")) {
        return None;
    }
    let positional: Vec<&crate::Parameter> =
        m.args.posonlyargs.iter().chain(m.args.args.iter()).collect();
    if positional.is_empty() {
        return None;
    }
    let defaulted_from = positional.len().saturating_sub(m.args.defaults.len());
    let rebound = rebound_names(&m.body);
    let mut out = Vec::new();
    for (i, p) in positional.iter().enumerate().skip(1) {
        let eligible = p.annotation.is_none() && i < defaulted_from && !rebound.contains(&p.arg);
        out.push((p.arg.clone(), eligible));
    }
    for (i, p) in m.args.kwonlyargs.iter().enumerate() {
        let defaulted = m.args.kw_defaults.get(i).is_some_and(|d| d.is_some());
        let eligible = p.annotation.is_none() && !defaulted && !rebound.contains(&p.arg);
        out.push((p.arg.clone(), eligible));
    }
    Some(out)
}

/// The names a body binds (assignment, loop or `with` target, augmented
/// store, `del`, a nested definition): a parameter rebound to another
/// value keeps its boxed slot.
pub(crate) fn rebound_names(body: &[Statement]) -> HashSet<String> {
    let mut out = HashSet::new();
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        // Only a NAME target rebinds (bare, or inside a destructuring): a
        // store THROUGH the parameter (`p.x = 1`, `p[k] = v`) mutates the
        // object it holds.
        fn bound(t: &ExprType, out: &mut HashSet<String>) {
            match t {
                ExprType::Name(n) => {
                    out.insert(n.id.clone());
                }
                ExprType::Tuple(tu) => tu.elts.iter().for_each(|e| bound(e, out)),
                ExprType::List(l) => l.iter().for_each(|e| bound(e, out)),
                ExprType::Starred(st) => bound(&st.value, out),
                _ => {}
            }
        }
        for t in visit::stmt_targets(s) {
            bound(t, &mut out);
        }
        match &s.statement {
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                out.insert(f.name.clone());
            }
            StatementType::ClassDef(c) => {
                out.insert(c.name.clone());
            }
            _ => {}
        }
        Flow::Continue
    });
    out
}

/// The topmost class of `class`'s chain that defines `method`.
fn slot_root(
    class: &ClassDef,
    method: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<String> {
    class
        .base_chain_with_options(symbols, options)
        .iter()
        .rev()
        .find(|c| c.methods().any(|m| m.name == method))
        .map(|c| c.name.clone())
}

/// Every function body of `stmts` with its enclosing class, when it is a
/// method (a `def` directly in a class body).
fn function_defs<'a>(
    stmts: &'a [Statement],
    class: Option<&'a str>,
    out: &mut Vec<(Option<&'a str>, &'a FunctionDef)>,
) {
    for s in stmts {
        match &s.statement {
            StatementType::ClassDef(c) => function_defs(&c.body, Some(&c.name), out),
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                out.push((class, f));
                function_defs(&f.body, None, out);
            }
            _ => {
                for body in visit::stmt_bodies(s) {
                    function_defs(body, class, out);
                }
            }
        }
    }
}

/// The evidence every call site of the crate gives, under the current
/// answers (`solved`).
fn collect(
    units: &[Unit],
    inferable: &HashMap<SlotKey, Vec<usize>>,
    slots: &BTreeMap<SlotKey, Vec<SlotDef>>,
    solved: &HashMap<(SlotKey, usize), String>,
    unambiguous: &dyn Fn(&str) -> bool,
) -> Collected {
    // The current answer for a method's parameter, by the defining class.
    let mut answer: HashMap<(String, String, String), String> = HashMap::new();
    for ((key, pos), class) in solved {
        for d in &slots[key] {
            answer.insert((d.class.clone(), key.1.clone(), d.params[*pos].0.clone()), class.clone());
        }
    }
    let mut out = Collected::default();
    for u in units {
        let module_name = u.options.this_module_path.last().cloned().unwrap_or_default();
        // The module's own statements, then every function body.
        let module_ctx = CodeGenContext::Module(module_name.clone());
        scan_body(&u.module.raw.body, None, &module_ctx, &u.symbols, &u.options, &HashSet::new(), inferable, unambiguous, &mut out);
        let mut defs = Vec::new();
        function_defs(&u.module.raw.body, None, &mut defs);
        for (class, f) in defs {
            let mut fsyms = u.symbols.clone();
            for s in &f.body {
                fsyms = s.clone().find_symbols(fsyms);
            }
            let is_method = class.is_some()
                && !f.decorator_list.iter().any(|d| {
                    matches!(d, ExprType::Name(n) if n.id == "staticmethod" || n.id == "classmethod")
                });
            let mut seeded: HashMap<String, TypeInfo> = HashMap::new();
            let mut pending: HashSet<String> = HashSet::new();
            let params: Vec<&crate::Parameter> = f
                .args
                .posonlyargs
                .iter()
                .chain(f.args.args.iter())
                .chain(f.args.kwonlyargs.iter())
                .collect();
            for (i, p) in params.iter().enumerate() {
                if is_method && i == 0 {
                    continue;
                }
                if let Some(ann) = p.annotation.as_deref() {
                    if let Some(t) = crate::resolve_alias_typeinfo(ann, &fsyms, &u.options)
                        .or_else(|| crate::annotation_type_info(ann))
                    {
                        seeded.insert(p.arg.clone(), t);
                    }
                } else if let Some(class) = class.filter(|_| is_method) {
                    match answer.get(&(class.to_string(), f.name.clone(), p.arg.clone())) {
                        Some(x) => {
                            seeded.insert(p.arg.clone(), TypeInfo::Class(x.clone()));
                        }
                        None => {
                            seeded.insert(p.arg.clone(), TypeInfo::PyValue);
                            pending.insert(p.arg.clone());
                        }
                    }
                }
            }
            let mut view = u.options.clone();
            view.name_types = Rc::new(seeded.clone());
            let info = crate::analyze_function_types_with_class(
                &f.body,
                Some(&view),
                Some(&fsyms),
                class.filter(|_| is_method),
            );
            let mut name_types = info.name_types;
            name_types.extend(seeded);
            view.name_types = Rc::new(name_types);
            let ctx = match class.filter(|_| is_method) {
                Some(c) => CodeGenContext::Class(c.to_string()),
                None => module_ctx.clone(),
            };
            scan_body(&f.body, class.filter(|_| is_method), &ctx, &fsyms, &view, &pending, inferable, unambiguous, &mut out);
        }
    }
    out
}

/// The calls of one body (its own scope: a nested definition is scanned
/// as its own body), recorded against the slots they resolve to.
#[allow(clippy::too_many_arguments)]
fn scan_body(
    body: &[Statement],
    _class: Option<&str>,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    view: &PythonOptions,
    pending: &HashSet<String>,
    inferable: &HashMap<SlotKey, Vec<usize>>,
    unambiguous: &dyn Fn(&str) -> bool,
    out: &mut Collected,
) {
    // The slot a `recv.name` resolves to, when it is a candidate.
    let resolve = |recv: &ExprType, name: &str| -> Option<(SlotKey, FunctionDef)> {
        let (class, class_symbols) = crate::ast::tree::call::receiver_class(recv, ctx, symbols, view)?;
        let def = class.method_on_mro_with_options(name, &class_symbols, view)?;
        let root = slot_root(&class, name, &class_symbols, view)?;
        let key = (root, name.to_string());
        inferable.contains_key(&key).then_some((key, def))
    };
    // A CLASS OBJECT as the receiver (`Base.m(obj, x)` — the unbound
    // call shifts the arguments by the explicit receiver).
    let class_object = |recv: &ExprType| -> Option<ClassDef> {
        let ExprType::Name(n) = recv else {
            return None;
        };
        if view.name_types.contains_key(&n.id) {
            return None;
        }
        crate::resolve_class_referenced(&n.id, symbols, view)
    };
    let evidence = |arg: &ExprType| -> Evidence {
        match crate::infer_type(Some(ctx), arg, view, symbols) {
            TypeInfo::Class(c) if unambiguous(&c) => {
                Evidence::Class(crate::ast::tree::hierarchy::canonical_class_name(&c, symbols))
            }
            _ => match arg {
                ExprType::Name(n) if pending.contains(&n.id) => Evidence::Pending,
                _ => Evidence::Other,
            },
        }
    };
    let mut called: HashSet<*const ExprType> = HashSet::new();
    let mut refs: Vec<(&ExprType, &str)> = Vec::new();
    visit::walk_stmts(body, Descend::SkipDefs, &mut |s| {
        if matches!(
            s.statement,
            StatementType::FunctionDef(_) | StatementType::AsyncFunctionDef(_) | StatementType::ClassDef(_)
        ) {
            return Flow::Skip;
        }
        for e in visit::stmt_exprs(s) {
            visit::walk_expr(e, &mut |e| match e {
                ExprType::Call(call) => {
                    let ExprType::Attribute(a) = call.func.as_ref() else {
                        return;
                    };
                    called.insert(call.func.as_ref() as *const ExprType);
                    if let Some(class) = class_object(&a.value) {
                        if let Some(root) = slot_root(&class, &a.attr, symbols, view) {
                            out.poisoned.insert((root, a.attr.clone()));
                        }
                        return;
                    }
                    let Some((key, def)) = resolve(&a.value, &a.attr) else {
                        return;
                    };
                    if call.args.iter().any(|x| matches!(x, ExprType::Starred(_)))
                        || call.keywords.iter().any(|k| k.arg.is_none())
                    {
                        out.poisoned.insert(key);
                        return;
                    }
                    let names: Vec<String> = instance_params(&def)
                        .map(|ps| ps.into_iter().map(|(n, _)| n).collect())
                        .unwrap_or_default();
                    let positions = &inferable[&key];
                    for (i, arg) in call.args.iter().enumerate() {
                        if positions.contains(&i) {
                            out.evidence.entry((key.clone(), i)).or_default().push(evidence(arg));
                        }
                    }
                    for kw in &call.keywords {
                        let Some(name) = kw.arg.as_deref() else {
                            continue;
                        };
                        if let Some(i) = names.iter().position(|n| n == name)
                            && positions.contains(&i)
                        {
                            out.evidence.entry((key.clone(), i)).or_default().push(evidence(&kw.value));
                        }
                    }
                }
                ExprType::Attribute(a) => refs.push((e, a.attr.as_str())),
                _ => {}
            });
        }
        Flow::Continue
    });
    // A method read as a VALUE (a bound method handed on): its callers
    // are out of sight.
    for (e, name) in refs {
        if called.contains(&(e as *const ExprType)) {
            continue;
        }
        let ExprType::Attribute(a) = e else { continue };
        if let Some((key, _)) = resolve(&a.value, name) {
            out.poisoned.insert(key);
        }
    }
}
