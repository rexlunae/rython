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
//! Consumers ask [`is_shared`] (the thread-local registry the module
//! installs) or `options.shared_classes`.

use std::collections::{BTreeMap, HashSet};

use crate::{ClassDef, ExprType, PythonOptions, Statement, StatementType};

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

/// The container annotations whose element (or value, for the mappings)
/// is a stored instance.
const SEQUENCE_CONTAINERS: &[&str] = &[
    "list", "List", "set", "Set", "frozenset", "FrozenSet", "Sequence", "MutableSequence",
    "Iterable", "Collection", "deque", "Deque",
];
const MAPPING_CONTAINERS: &[&str] = &[
    "dict", "Dict", "Mapping", "MutableMapping", "OrderedDict", "defaultdict", "DefaultDict",
];

/// Compute the shared set: `this_body` and `this_classes` are the module
/// being converted; every other module comes from `options.module_defs`
/// (its emitted classes, as the hierarchy index sees them).
pub fn compute_shared(
    this_body: &[Statement],
    this_classes: &[ClassDef],
    options: &PythonOptions,
) -> HashSet<String> {
    let mut classes: BTreeMap<String, ClassDef> = BTreeMap::new();
    let mut stored: HashSet<String> = HashSet::new();
    let mut external_store_fields: HashSet<String> = HashSet::new();
    let mut register = |body: &[Statement], defs: Vec<ClassDef>| {
        for c in defs {
            classes.entry(c.name.clone()).or_insert(c);
        }
        collect_container_elements(body, &mut stored);
        collect_external_store_fields(body, &mut external_store_fields);
    };
    register(this_body, this_classes.to_vec());
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
        register(&module.raw.body, defs);
    }
    let mut shared: HashSet<String> = classes
        .iter()
        .filter(|(name, c)| {
            stored.contains(*name)
                && !crate::ast::tree::class_def::is_exception_class(c)
                && (has_mutating_method(c)
                    || own_field_names(c).iter().any(|f| external_store_fields.contains(f)))
        })
        .map(|(name, _)| name.clone())
        .collect();
    // Family closure: a root's sum type holds its members, so the root
    // and every class of its subtree take the one representation.
    let roots = options.hierarchy_roots.clone();
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

/// The class names in element position of a container annotation, over
/// every annotation in `stmts` (parameters, returns, annotated stores),
/// recursing through nested containers and every statement body.
fn collect_container_elements(stmts: &[Statement], out: &mut HashSet<String>) {
    fn from_annotation(ann: &ExprType, in_container: bool, out: &mut HashSet<String>) {
        // A string annotation re-parses like the annotation authority.
        let unquoted = crate::ast::tree::arguments::unquote_annotation(ann);
        let ann: &ExprType = unquoted.as_ref().unwrap_or(ann);
        match ann {
            ExprType::Name(n) => {
                if in_container {
                    out.insert(n.id.clone());
                }
            }
            ExprType::Subscript(sub) => {
                let head = match sub.value.as_ref() {
                    ExprType::Name(n) => Some(n.id.as_str()),
                    ExprType::Attribute(a) => Some(a.attr.as_str()),
                    _ => None,
                };
                let crate::SubscriptKind::Index(index) = &sub.kind else {
                    return;
                };
                let elts: Vec<&ExprType> = match index.as_ref() {
                    ExprType::Tuple(t) => t.elts.iter().collect(),
                    other => vec![other],
                };
                match head {
                    Some(h) if SEQUENCE_CONTAINERS.contains(&h) => {
                        for e in elts {
                            from_annotation(e, true, out);
                        }
                    }
                    Some(h) if MAPPING_CONTAINERS.contains(&h) => {
                        if let Some(v) = elts.last() {
                            from_annotation(v, true, out);
                        }
                    }
                    // `Optional[T]`, `tuple[...]`, `type[T]`, ...: the
                    // element keeps the enclosing container's status.
                    _ => {
                        for e in elts {
                            from_annotation(e, in_container, out);
                        }
                    }
                }
            }
            // `T | None`, `list[A] | list[B]`.
            ExprType::BinOp(b) => {
                from_annotation(&b.left, in_container, out);
                from_annotation(&b.right, in_container, out);
            }
            _ => {}
        }
    }
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
                        from_annotation(a, false, out);
                    }
                }
                if let Some(r) = f.returns.as_deref() {
                    from_annotation(r, false, out);
                }
                collect_container_elements(&f.body, out);
            }
            StatementType::ClassDef(c) => collect_container_elements(&c.body, out),
            StatementType::Assign(a) => {
                if let Some(ann) = a.annotation.as_ref() {
                    from_annotation(ann, false, out);
                }
            }
            StatementType::AnnotatedName { annotation, .. } => {
                from_annotation(annotation, false, out);
            }
            StatementType::If(i) => {
                collect_container_elements(&i.body, out);
                collect_container_elements(&i.orelse, out);
            }
            StatementType::For(f) => {
                collect_container_elements(&f.body, out);
                collect_container_elements(&f.orelse, out);
            }
            StatementType::While(w) => {
                collect_container_elements(&w.body, out);
                collect_container_elements(&w.orelse, out);
            }
            StatementType::With(w) => collect_container_elements(&w.body, out),
            StatementType::Try(t) => {
                collect_container_elements(&t.body, out);
                for h in &t.handlers {
                    collect_container_elements(&h.body, out);
                }
                collect_container_elements(&t.orelse, out);
                collect_container_elements(&t.finalbody, out);
            }
            _ => {}
        }
    }
}

/// The attribute names stored through a NON-`self` receiver anywhere
/// (`acct.balance = 1`, `item.qty -= qty`): a class with such a field is
/// mutated from outside its own methods.
fn collect_external_store_fields(stmts: &[Statement], out: &mut HashSet<String>) {
    fn target(t: &ExprType, out: &mut HashSet<String>) {
        match t {
            ExprType::Attribute(a) => {
                if !matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") {
                    out.insert(a.attr.clone());
                }
            }
            ExprType::Tuple(tu) => tu.elts.iter().for_each(|e| target(e, out)),
            ExprType::List(l) => l.iter().for_each(|e| target(e, out)),
            _ => {}
        }
    }
    for s in stmts {
        match &s.statement {
            StatementType::Assign(a) => a.targets.iter().for_each(|t| target(t, out)),
            StatementType::AugAssign(a) => target(&a.target, out),
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                collect_external_store_fields(&f.body, out)
            }
            StatementType::ClassDef(c) => collect_external_store_fields(&c.body, out),
            StatementType::If(i) => {
                collect_external_store_fields(&i.body, out);
                collect_external_store_fields(&i.orelse, out);
            }
            StatementType::For(f) => {
                collect_external_store_fields(&f.body, out);
                collect_external_store_fields(&f.orelse, out);
            }
            StatementType::While(w) => {
                collect_external_store_fields(&w.body, out);
                collect_external_store_fields(&w.orelse, out);
            }
            StatementType::With(w) => collect_external_store_fields(&w.body, out),
            StatementType::Try(t) => {
                collect_external_store_fields(&t.body, out);
                for h in &t.handlers {
                    collect_external_store_fields(&h.body, out);
                }
                collect_external_store_fields(&t.orelse, out);
                collect_external_store_fields(&t.finalbody, out);
            }
            _ => {}
        }
    }
}

/// The field names a class stores through `self` in any of its methods.
fn own_field_names(c: &ClassDef) -> HashSet<String> {
    let mut out = HashSet::new();
    for m in c.methods() {
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
    for m in c.methods() {
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
    use crate::ast::tree::scope::CONTAINER_MUTATING_METHODS;
    fn is_self(e: &ExprType) -> bool {
        matches!(e, ExprType::Name(n) if n.id == "self")
    }
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
        if let ExprType::Call(c) = e {
            if let ExprType::Attribute(a) = c.func.as_ref() {
                if is_self(&a.value) {
                    self_calls.push(a.attr.clone());
                } else if let Some(f) = self_field(&a.value)
                    && CONTAINER_MUTATING_METHODS.contains(&a.attr.as_str())
                {
                    fields.push(f);
                    found = true;
                }
            }
            for a in c.args.iter().chain(c.keywords.iter().map(|k| &k.value)) {
                found |= expr_mutates(a, fields, self_calls);
            }
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
                found |= expr_mutates(&a.value, fields, self_calls);
            }
            StatementType::AugAssign(a) => {
                if let Some(f) = self_field(&a.target) {
                    fields.push(f);
                    found = true;
                }
            }
            StatementType::Expr(e) => found |= expr_mutates(&e.value, fields, self_calls),
            StatementType::Return(Some(e)) => found |= expr_mutates(&e.value, fields, self_calls),
            StatementType::If(i) => {
                found |= self_stores(&i.body, fields, self_calls);
                found |= self_stores(&i.orelse, fields, self_calls);
            }
            StatementType::For(f) => {
                found |= self_stores(&f.body, fields, self_calls);
                found |= self_stores(&f.orelse, fields, self_calls);
            }
            StatementType::While(w) => {
                found |= self_stores(&w.body, fields, self_calls);
                found |= self_stores(&w.orelse, fields, self_calls);
            }
            StatementType::With(w) => found |= self_stores(&w.body, fields, self_calls),
            StatementType::Try(t) => {
                found |= self_stores(&t.body, fields, self_calls);
                for h in &t.handlers {
                    found |= self_stores(&h.body, fields, self_calls);
                }
                found |= self_stores(&t.orelse, fields, self_calls);
                found |= self_stores(&t.finalbody, fields, self_calls);
            }
            _ => {}
        }
    }
    found
}
