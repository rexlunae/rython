//! The SHARED classes of the crate (issue #137, the aliasing
//! representation): a class whose instances are stored in a container
//! anywhere in the crate AND mutated after construction anywhere in the
//! crate holds its state behind `stdpython::PyRef<T>` (`Rc<RefCell<T>>`),
//! so a local fetched from the container, the container slot, and every
//! other holder are ONE object — CPython's reference semantics for the
//! shape that would otherwise diverge silently (`item = self.find(name)`;
//! `item.qty -= qty`; `acct.deposit(5)`). Every other class stays a plain
//! struct: cloning an immutable object, or one that no container holds,
//! is unobservable.
//!
//! The set is computed once per module conversion over every module of
//! the crate (the same crate-wide walk the hierarchy index takes) and
//! closed over hierarchy families: a root and its subtree share one
//! representation, since the root's sum type holds the members.
//! Consumers ask [`is_shared`] — the one registry, installed per module
//! conversion like the hierarchy index (`hierarchy::install_roots`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::tree::visit::{is_self, stmt_bodies, stmt_exprs, subexprs};
use crate::{ClassDef, CodeGen, ExprType, FunctionDef, PythonOptions, Statement, StatementType, SymbolTableScopes, TypeInfo};

thread_local! {
    static SHARED: std::cell::RefCell<std::rc::Rc<HashSet<String>>> =
        std::cell::RefCell::new(std::rc::Rc::new(HashSet::new()));
}

/// Whether `name` is a shared class (see the module doc).
pub fn is_shared(name: &str) -> bool {
    SHARED.with(|s| s.borrow().contains(name))
}

/// Install the registry for the module being converted.
pub fn install_shared(shared: &HashSet<String>) {
    SHARED.with(|s| *s.borrow_mut() = std::rc::Rc::new(shared.clone()));
}

/// Compute the shared set: `this_body` and `this_classes` are the module
/// being converted (with its symbols); every other module comes from
/// `options.module_defs` (its emitted classes, as the hierarchy index
/// sees them).
pub fn compute_shared(
    this_body: &[Statement],
    this_classes: &[ClassDef],
    this_symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> HashSet<String> {
    let mut classes: BTreeMap<String, ClassDef> = BTreeMap::new();
    // How many emitted modules define each class name: the registry is
    // keyed by the bare name the type side carries, so a name two modules
    // both define is AMBIGUOUS — excluded from sharing, loudly (the
    // hierarchy index applies the same rule; Devin review on #321).
    let mut defined_in: HashMap<String, usize> = HashMap::new();
    let mut stored: HashSet<String> = HashSet::new();
    let mut external_stores: HashSet<ExternalStore> = HashSet::new();
    let mut register = |body: &[Statement], defs: Vec<ClassDef>, symbols: &SymbolTableScopes, opts: &PythonOptions| {
        for c in defs {
            *defined_in.entry(c.name.clone()).or_insert(0) += 1;
            classes.entry(c.name.clone()).or_insert(c);
        }
        collect_container_elements(body, symbols, opts, &mut stored);
        collect_external_store_fields(body, &Env::default(), symbols, opts, &mut external_stores);
    };
    register(this_body, this_classes.to_vec(), this_symbols, options);
    for (path, module) in options.module_defs.iter() {
        if path[..] == options.this_module_path[..] {
            continue;
        }
        let mut module_opts = options.clone();
        let is_package = options
            .module_defs
            .keys()
            .any(|k| k.len() > path.len() && k[..path.len()] == path[..]);
        module_opts.module_path = if is_package {
            path.clone()
        } else {
            path[..path.len().saturating_sub(1)].to_vec()
        };
        module_opts.this_module_path = path.clone();
        let defs = crate::ast::tree::module::emitted_class_defs(module, &module_opts);
        let module: &crate::Module = module;
        let module_symbols = module.clone().find_symbols(SymbolTableScopes::new());
        register(&module.raw.body, defs, &module_symbols, &module_opts);
    }
    // Mutability is INHERITED: a stored subclass whose only mutator is
    // its base's is mutated through it (Devin review on #321). And both
    // facts are FAMILY facts: a root-typed container (`shapes:
    // list[Shape]`) stores the subtree's instances, and a subclass's
    // mutator (`Rect.scale`) mutates them — the store is recorded under
    // the root and the mutation under the subclass, so each is asked of
    // the whole family (the evaluation on issue #137: `views`'s Rect,
    // stored under Shape and mutated through a parameter, stayed a
    // value, and a mutation through the parameter was silently lost).
    let roots = options.hierarchy_roots.clone();
    let family_of = |name: &str| -> Vec<String> {
        roots
            .iter()
            .find(|(root, subtree)| {
                root.as_str() == name || subtree.iter().any(|v| v.name == name)
            })
            .map(|(root, subtree)| {
                std::iter::once(root.clone())
                    .chain(subtree.iter().map(|v| v.name.clone()))
                    .collect()
            })
            .unwrap_or_else(|| vec![name.to_string()])
    };
    let mut memo: HashMap<String, bool> = HashMap::new();
    let names: Vec<String> = classes.keys().cloned().collect();
    let mut shared: HashSet<String> = names
        .iter()
        .filter(|name| {
            let c = &classes[*name];
            let family = family_of(name);
            let qualifies = family.iter().any(|m| stored.contains(m))
                && !crate::ast::tree::class_def::is_exception_class(c)
                && family
                    .iter()
                    .any(|m| class_mutates(m, &classes, &external_stores, &mut memo));
            if qualifies && defined_in.get(*name).copied().unwrap_or(0) > 1 {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}` is defined by more than one module of the crate: its \
                     instances stay values (sharing is keyed by the class name), so a \
                     mutation through a container-fetched alias of it does not reach \
                     the stored object (issue #137)",
                    name
                ));
                return false;
            }
            qualifies
        })
        .cloned()
        .collect();
    // Family closure: a root's sum type holds its members, so the root
    // and every class of its subtree take the one representation.
    let mut changed = true;
    while changed {
        changed = false;
        for (root, subtree) in roots.iter() {
            let family: Vec<&str> = std::iter::once(root.as_str())
                .chain(subtree.iter().map(|v| v.name.as_str()))
                .collect();
            if family.iter().any(|n| shared.contains(*n)) {
                for n in family {
                    if shared.insert(n.to_string()) {
                        changed = true;
                    }
                }
            }
        }
    }
    shared
}

/// The classes held in element position of a container type anywhere in
/// `stmts` — a parameter, return, or annotated store (through the
/// alias-aware annotation authority, so `Items = list[Item]` counts), an
/// un-annotated store typed by the inferrer (`{"x": Item()}`), and every
/// class's inferred field table (the stores its methods make) — recursing
/// through nested containers and every statement body. What is a
/// container is what `TypeInfo` renders as one: the boxed generics
/// (`Sequence[T]`, `Iterable[T]`) hold no struct, so they hold no
/// instance to share.
fn collect_container_elements(
    stmts: &[Statement],
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    out: &mut HashSet<String>,
) {
    fn from_type(t: &TypeInfo, in_container: bool, out: &mut HashSet<String>) {
        match t {
            TypeInfo::Vec(inner) | TypeInfo::HashSet(inner) => from_type(inner, true, out),
            TypeInfo::Dict(_, v) => from_type(v, true, out),
            // A tuple HOLDS its elements as a list does (`pair: tuple[Item,
            // Item]`; Devin review on #321).
            TypeInfo::Tuple(items) => items.iter().for_each(|i| from_type(i, true, out)),
            TypeInfo::Option(inner) => from_type(inner, in_container, out),
            TypeInfo::Class(c) => {
                if in_container {
                    out.insert(c.clone());
                }
            }
            _ => {}
        }
    }
    let from_annotation = |ann: &ExprType, out: &mut HashSet<String>| {
        if let Some(t) = crate::resolve_alias_typeinfo(ann, symbols, options)
            .or_else(|| crate::annotation_type_info(ann))
        {
            from_type(&t, false, out);
        }
    };
    for s in stmts {
        match &s.statement {
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                for p in f
                    .args
                    .posonlyargs
                    .iter()
                    .chain(f.args.args.iter())
                    .chain(f.args.kwonlyargs.iter())
                {
                    if let Some(a) = p.annotation.as_deref() {
                        from_annotation(a, out);
                    }
                }
                if let Some(r) = f.returns.as_deref() {
                    from_annotation(r, out);
                }
                collect_container_elements(&f.body, symbols, options, out);
            }
            StatementType::ClassDef(c) => {
                if let Ok(fields) = c.infer_fields(symbols, options) {
                    for (_, t) in &fields {
                        from_type(t, false, out);
                    }
                }
                collect_container_elements(&c.body, symbols, options, out);
            }
            StatementType::Assign(a) => match a.annotation.as_ref() {
                Some(ann) => from_annotation(ann, out),
                None => from_type(&crate::infer_type(None, &a.value, options, symbols), false, out),
            },
            StatementType::AnnotatedName { annotation, .. } => {
                from_annotation(annotation, out);
            }
            _ => {
                for body in stmt_bodies(s) {
                    collect_container_elements(body, symbols, options, out);
                }
            }
        }
    }
}

/// A store through a NON-`self` receiver (`acct.balance = 1`,
/// `item.qty -= qty`): the field, with the receiver's class when the
/// scope names it — an annotated parameter, a local constructed from a
/// class, an element of an annotated container — and `None` when it
/// does not (the store then counts for every class with that field: the
/// over-approximation errs toward sharing, which is exact for every
/// alias shape but the one the spec names; Devin review on #321).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExternalStore {
    field: String,
    receiver_class: Option<String>,
}

impl ExternalStore {
    /// Whether this store mutates `class`'s field `field`.
    fn hits(&self, class: &str, field: &str) -> bool {
        self.field == field && self.receiver_class.as_deref().is_none_or(|c| c == class)
    }
}

/// The names a function's scope types as CLASS INSTANCES (a parameter's
/// annotation, a local's construction or annotation, an element of an
/// annotated container), for the receiver of an external store.
#[derive(Clone, Default)]
struct Env {
    names: HashMap<String, TypeInfo>,
}

impl Env {
    fn class_of(&self, receiver: &ExprType) -> Option<String> {
        let element = |t: &TypeInfo| -> Option<String> {
            match t {
                TypeInfo::Vec(inner) | TypeInfo::HashSet(inner) => match inner.as_ref() {
                    TypeInfo::Class(c) => Some(c.clone()),
                    _ => None,
                },
                TypeInfo::Dict(_, v) => match v.as_ref() {
                    TypeInfo::Class(c) => Some(c.clone()),
                    _ => None,
                },
                _ => None,
            }
        };
        match receiver {
            ExprType::Name(n) => match self.names.get(&n.id)? {
                TypeInfo::Class(c) => Some(c.clone()),
                TypeInfo::Option(inner) => match inner.as_ref() {
                    TypeInfo::Class(c) => Some(c.clone()),
                    _ => None,
                },
                _ => None,
            },
            ExprType::Subscript(sub) => match sub.value.as_ref() {
                ExprType::Name(n) => element(self.names.get(&n.id)?),
                _ => None,
            },
            _ => None,
        }
    }
}

fn collect_external_store_fields(
    stmts: &[Statement],
    env: &Env,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    out: &mut HashSet<ExternalStore>,
) {
    fn target(t: &ExprType, env: &Env, out: &mut HashSet<ExternalStore>) {
        match t {
            ExprType::Attribute(a) => {
                if !is_self(a.value.as_ref()) {
                    out.insert(ExternalStore {
                        field: a.attr.clone(),
                        receiver_class: env.class_of(&a.value),
                    });
                }
            }
            ExprType::Tuple(tu) => tu.elts.iter().for_each(|e| target(e, env, out)),
            ExprType::List(l) => l.iter().for_each(|e| target(e, env, out)),
            _ => {}
        }
    }
    // A class name through its aliases (`Dial = Knob`, `from m import Knob
    // as K`): the registry's name, so the store keys to the class the
    // mutability decision knows (Devin review on #321).
    let canonical = |t: TypeInfo| -> TypeInfo {
        match t {
            TypeInfo::Class(c) => {
                TypeInfo::Class(crate::ast::tree::hierarchy::canonical_class_name(&c, symbols))
            }
            TypeInfo::Option(inner) => match *inner {
                TypeInfo::Class(c) => TypeInfo::Option(Box::new(TypeInfo::Class(
                    crate::ast::tree::hierarchy::canonical_class_name(&c, symbols),
                ))),
                other => TypeInfo::Option(Box::new(other)),
            },
            other => other,
        }
    };
    let annotated = |ann: &ExprType| -> Option<TypeInfo> {
        crate::annotation_type_info(ann)
            .or_else(|| crate::resolve_alias_typeinfo(ann, symbols, options))
            .map(canonical)
    };
    // A function opens a scope: its annotated parameters, then the locals
    // its body constructs or annotates.
    let function_env = |f: &FunctionDef| -> Env {
        let mut env = env.clone();
        for p in f.args.posonlyargs.iter().chain(f.args.args.iter()).chain(f.args.kwonlyargs.iter()) {
            if let Some(ann) = p.annotation.as_deref()
                && let Some(t) = annotated(ann)
            {
                env.names.insert(p.arg.clone(), t);
            }
        }
        env
    };
    // A container-mutating call on a FIELD reached through a non-`self`
    // receiver (`item.values.append(x)`, `item.mapping.update(..)`): the
    // field's object mutates — the same mutator authority the scope
    // analysis and the call lowering use (Devin review on #321).
    fn mutating_calls(e: &ExprType, env: &Env, out: &mut HashSet<ExternalStore>) {
        if let ExprType::Call(c) = e
            && let ExprType::Attribute(method) = c.func.as_ref()
            && crate::ast::tree::scope::mutates_receiver(&method.attr)
            && let ExprType::Attribute(field) = method.value.as_ref()
            && !is_self(field.value.as_ref())
        {
            out.insert(ExternalStore {
                field: field.attr.clone(),
                receiver_class: env.class_of(&field.value),
            });
        }
        for sub in subexprs(e) {
            mutating_calls(sub, env, out);
        }
    }
    let mut env = env.clone();
    for s in stmts {
        for e in stmt_exprs(s) {
            mutating_calls(e, &env, out);
        }
        match &s.statement {
            StatementType::Assign(a) => {
                a.targets.iter().for_each(|t| target(t, &env, out));
                if let [ExprType::Name(n)] = a.targets.as_slice() {
                    let typed = a
                        .annotation
                        .as_ref()
                        .and_then(|ann| annotated(ann))
                        .or_else(|| match &a.value {
                            ExprType::Call(c) => match c.func.as_ref() {
                                ExprType::Name(callee) => {
                                    crate::resolve_class_referenced(&callee.id, symbols, options)
                                        .map(|class| TypeInfo::Class(class.name))
                                }
                                _ => None,
                            },
                            _ => None,
                        });
                    match typed {
                        Some(t) => {
                            env.names.insert(n.id.clone(), t);
                        }
                        None => {
                            env.names.remove(&n.id);
                        }
                    }
                }
            }
            StatementType::AugAssign(a) => target(&a.target, &env, out),
            StatementType::Delete(targets) => targets.iter().for_each(|t| target(t, &env, out)),
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                collect_external_store_fields(&f.body, &function_env(f), symbols, options, out);
                continue;
            }
            _ => {}
        }
        for body in stmt_bodies(s) {
            collect_external_store_fields(body, &env, symbols, options, out);
        }
    }
}

/// Every method of the class, the asynchronous ones included (a mutation
/// in an `async def` is a mutation), overloads excluded as `methods()`
/// excludes them.
fn all_methods(c: &ClassDef) -> impl Iterator<Item = &FunctionDef> {
    c.body.iter().filter_map(|s| match &s.statement {
        StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
            let is_overload = f.decorator_list.iter().any(|d| match d {
                ExprType::Name(n) => n.id == "overload",
                ExprType::Attribute(a) => a.attr == "overload",
                _ => false,
            });
            if is_overload { None } else { Some(f) }
        }
        _ => None,
    })
}

/// Whether the class is mutated after construction: its own methods (see
/// `has_mutating_method`), a field of its own stored from outside, or —
/// inheritance — the same of any base in the crate.
fn class_mutates(
    name: &str,
    classes: &BTreeMap<String, ClassDef>,
    external_stores: &HashSet<ExternalStore>,
    memo: &mut HashMap<String, bool>,
) -> bool {
    if let Some(&m) = memo.get(name) {
        return m;
    }
    // A cycle (or an unknown base) is not a mutation.
    memo.insert(name.to_string(), false);
    let Some(c) = classes.get(name) else {
        return false;
    };
    let own = has_mutating_method(c)
        || own_field_names(c)
            .iter()
            .any(|f| external_stores.iter().any(|st| st.hits(name, f)));
    let inherited = c.bases.iter().any(|b| match b {
        ExprType::Name(n) => class_mutates(&n.id, classes, external_stores, memo),
        ExprType::Attribute(a) => class_mutates(&a.attr, classes, external_stores, memo),
        _ => false,
    });
    let result = own || inherited;
    memo.insert(name.to_string(), result);
    result
}

/// The field names a class stores through `self` in any of its methods.
fn own_field_names(c: &ClassDef) -> HashSet<String> {
    let mut out = HashSet::new();
    for m in all_methods(c) {
        let mut stores = Vec::new();
        self_stores(&m.body, &mut stores, &mut Vec::new());
        out.extend(stores);
    }
    out
}

/// Whether any method other than `__init__` mutates `self`: a store or
/// augmented store into a `self` field, a container-mutating call on
/// one, a `del`, or a call to another such method of the class.
fn has_mutating_method(c: &ClassDef) -> bool {
    let mut direct: HashSet<String> = HashSet::new();
    let mut calls: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for m in all_methods(c) {
        if m.name == "__init__" {
            continue;
        }
        let mut stores = Vec::new();
        let mut self_calls = Vec::new();
        if self_stores(&m.body, &mut stores, &mut self_calls) {
            direct.insert(m.name.clone());
        }
        calls.insert(m.name.clone(), self_calls);
    }
    if !direct.is_empty() {
        // Transitive through same-class calls only decides WHICH methods
        // mutate; any direct mutation already makes the class mutable.
        return true;
    }
    let _ = calls;
    false
}

/// Whether `stmts` mutate `self` (see `has_mutating_method`); collects the
/// stored field names and the `self.<method>()` callees on the way.
fn self_stores(stmts: &[Statement], fields: &mut Vec<String>, self_calls: &mut Vec<String>) -> bool {
    fn self_field(t: &ExprType) -> Option<String> {
        match t {
            ExprType::Attribute(a) if is_self(&a.value) => Some(a.attr.clone()),
            // `self.f[k] = v`, `self.f.g = v`: the field `f` is mutated.
            ExprType::Attribute(a) => self_field(&a.value),
            ExprType::Subscript(s) => self_field(&s.value),
            _ => None,
        }
    }
    fn expr_mutates(e: &ExprType, fields: &mut Vec<String>, self_calls: &mut Vec<String>) -> bool {
        let mut found = false;
        if let ExprType::Call(c) = e
            && let ExprType::Attribute(a) = c.func.as_ref()
        {
            if is_self(&a.value) {
                self_calls.push(a.attr.clone());
            } else if let Some(f) = self_field(&a.value)
                && crate::ast::tree::scope::mutates_receiver(&a.attr)
            {
                fields.push(f);
                found = true;
            }
        }
        for sub in subexprs(e) {
            found |= expr_mutates(sub, fields, self_calls);
        }
        found
    }
    let mut found = false;
    for s in stmts {
        match &s.statement {
            StatementType::Assign(a) => {
                for t in &a.targets {
                    if let Some(f) = self_field(t) {
                        fields.push(f);
                        found = true;
                    }
                }
            }
            StatementType::AugAssign(a) => {
                if let Some(f) = self_field(&a.target) {
                    fields.push(f);
                    found = true;
                }
            }
            // `del self.items[i]`, `del self.cache`: a mutation of the
            // field (Devin review on #321).
            StatementType::Delete(targets) => {
                for t in targets {
                    if let Some(f) = self_field(t) {
                        fields.push(f);
                        found = true;
                    }
                }
            }
            // `for self.x in ..`, `with .. as self.x`: stores too.
            StatementType::For(f) => {
                if let Some(f) = self_field(&f.target) {
                    fields.push(f);
                    found = true;
                }
            }
            StatementType::AsyncFor(f) => {
                if let Some(f) = self_field(&f.target) {
                    fields.push(f);
                    found = true;
                }
            }
            StatementType::With(w) => {
                for item in &w.items {
                    if let Some(f) = item.optional_vars.as_ref().and_then(self_field) {
                        fields.push(f);
                        found = true;
                    }
                }
            }
            StatementType::AsyncWith(w) => {
                for item in &w.items {
                    if let Some(f) = item.optional_vars.as_ref().and_then(self_field) {
                        fields.push(f);
                        found = true;
                    }
                }
            }
            _ => {}
        }
        // The expressions the statement itself evaluates (a call in an
        // `if` test — Devin review on #321), then its bodies.
        for e in stmt_exprs(s) {
            found |= expr_mutates(e, fields, self_calls);
        }
        for body in stmt_bodies(s) {
            found |= self_stores(body, fields, self_calls);
        }
    }
    found
}
